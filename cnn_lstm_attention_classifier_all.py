import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

class ActivityDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class Attention(nn.Module):
    """Simple Self-Attention Mechanism"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        # Calculate attention scores (batch_size, seq_len, 1)
        scores = torch.matmul(lstm_out, self.attention_weights)  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum of LSTM outputs
        weighted_sum = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size * 2)
        
        return weighted_sum, attention_weights

class CNNLSTMWithAttention(nn.Module):
    def __init__(self, input_size=3, cnn_out_channels=256, kernel_size=3, lstm_hidden_size=256, num_lstm_layers=3, num_classes=10, dropout=0.2):
        super(CNNLSTMWithAttention, self).__init__()

        # CNN 部分
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels * 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels * 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM 部分
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels * 2,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention 部分
        self.attention = Attention(lstm_hidden_size * 2)  # 双向LSTM，输入的hidden_size * 2

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # CNN 部分
        x = x.transpose(1, 2)  # (batch_size, channels, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)

        # LSTM 部分
        lstm_out, _ = self.lstm(x)

        # Attention 部分
        attention_out, attention_weights = self.attention(lstm_out)

        # 全连接层
        out = self.fc(attention_out)
        return out, attention_weights  # 返回预测和 attention weights




def load_and_preprocess_data(base_dir, window_size=128):
    """加载和预处理数据"""
    sequences = []
    labels = []
    label_counts = {}  # 用来统计每个标签的序列数量

    
    # 遍历所有活动类型文件夹
    for activity_id in range(2806, 2816):  # 2806-2815
        activity_dir = os.path.join(base_dir, str(activity_id))
        if not os.path.exists(activity_dir):
            continue
            
        print(f"Processing activity {activity_id}...")
        
        # 读取该活动类型的所有CSV文件
        for file in os.listdir(activity_dir):
            if not file.endswith('.csv'):
                continue
                
            file_path = os.path.join(activity_dir, file)
            try:
                # 读取CSV文件
                data = pd.read_csv(
                    file_path,
                    header=None,
                    names=['RandomID', 'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
                )
                
                # 提取加速度数据
                sensor_data = data[['Accel_X', 'Accel_Y', 'Accel_Z']].values
                
                # 使用滑动窗口分割数据
                for i in range(0, len(sensor_data) - window_size + 1, window_size // 2):
                    sequence = sensor_data[i:i + window_size]
                    if len(sequence) == window_size:
                        sequences.append(sequence)
                        labels.append(activity_id - 2806)  # 将活动ID映射到0-9
                        label=activity_id - 2806

                        # 更新标签计数
                        if label not in label_counts:
                            label_counts[label] = 0
                        label_counts[label] += 1
                        
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue


    # 打印每个标签的序列数量
    print("Label counts (number of sequences for each label):")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} sequences")

    return np.array(sequences), np.array(labels)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    """Train and evaluate the model"""
    import numpy as np
    from sklearn.metrics import classification_report

    model = model.to(device)
    best_val_acc = 0
    best_model = None  # Save the best model weights

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, _ = model(sequences)  # Obtain predictions, ignore attention_weights
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs, _ = model(sequences)  # Obtain predictions, ignore attention_weights
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        # Display Training Info
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Save the best model if performance improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()  # Keep a copy of the best model's weights
            torch.save(best_model, 'best_cnn_lstm_attention_model.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')

        print('-' * 60)

    # Load the best model weights for evaluation
    print("Loading the best model for evaluation...")
    model.load_state_dict(torch.load('best_cnn_lstm_attention_model.pth'))

    # Final Evaluation Using Best Model
    print("Evaluating the best model's performance on validation data...")
    model.eval()

    all_labels_best_model = []
    all_predictions_best_model = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs, _ = model(sequences)  # Predictions from the loaded best model
            _, predicted = outputs.max(1)

            all_labels_best_model.extend(labels.cpu().numpy())
            all_predictions_best_model.extend(predicted.cpu().numpy())

    # Generate a classification report and print it
    print("\nBest model performance on each class:")
    print(classification_report(all_labels_best_model, all_predictions_best_model))

    # Calculate and print overall accuracy
    val_acc_final = (np.array(all_predictions_best_model) == np.array(all_labels_best_model)).mean() * 100
    print(f'\nBest model validation accuracy after loading: {val_acc_final:.2f}%')


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    sequences, labels = load_and_preprocess_data('activity_segments', window_size=128)
    
    # 标准化数据
    scaler = StandardScaler()
    sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences = sequences_scaled.reshape(sequences.shape)
    
    # 保存标准化器
    joblib.dump(scaler, 'cnn_lstms_attention_classifier_all_scaler.pkl')
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 创建数据加载器
    train_dataset = ActivityDataset(X_train, y_train)
    test_dataset = ActivityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = CNNLSTMWithAttention(
        input_size=3,          # 三轴加速度数据
        lstm_hidden_size=128,        # LSTM隐藏层大小
        num_lstm_layers=2,          # LSTM层数
        num_classes=10         # 10种活动类型
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=250,
        device=device
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 

