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


class CNNTransformerLSTMClassifier(nn.Module):
    def __init__(self, input_size=3, cnn_out_channels=64, kernel_size=3, lstm_hidden_size=128, num_lstm_layers=3, num_classes=10, dropout=0.2, transformer_d_model=128, transformer_nhead=4, transformer_num_layers=3):
        super(CNNTransformerLSTMClassifier, self).__init__()

        # CNN 部分
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels * 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels * 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Transformer 部分
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=transformer_nhead)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=transformer_num_layers)

        # LSTM 部分
        self.lstm = nn.LSTM(
            input_size=transformer_d_model,  
            hidden_size=lstm_hidden_size,     
            num_layers=num_lstm_layers,       
            batch_first=True,                  
            dropout=dropout,
            bidirectional=True                 
        )

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
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  

        # Transformer 部分
        transformer_out = self.transformer(x)

        # LSTM 部分
        lstm_out, _ = self.lstm(transformer_out)

        # 使用最后一个时间步的输出
        out = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        return out

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
    """训练模型"""
    model = model.to(device)
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)

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
    joblib.dump(scaler, 'lstm_scaler.pkl')
    
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
    model = CNNTransformerLSTMClassifier(
        input_size=3,          
        lstm_hidden_size=128,        
        num_lstm_layers=2,          
        num_classes=10,         
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
        num_epochs=500,
        device=device
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 

