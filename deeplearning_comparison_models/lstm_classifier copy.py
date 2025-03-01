import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from sklearn.metrics import classification_report

class ActivityDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    """LSTM分类模型"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=10):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        return out

def load_and_preprocess_data(base_dir, window_size=64):
    """加载和预处理数据"""
    sequences = []
    labels = []
    
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
                        
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
    
    return np.array(sequences), np.array(labels)




def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    """训练 DeepConvLSTM 模型"""
    model_name = 'deepconvlstm'  # 模型名称
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
        print(f'Epoch [{epoch+1}/{num_epochs}] ({model_name})')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}% ({model_name})')

        print('-' * 60)

    # 加载最佳模型进行评估
    print(f"加载最佳模型用于评估 ({model_name}) ...")
    model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))

    # 最终评估加载后的最佳模型在验证集上的表现
    print(f"正在评估加载后的最佳模型在验证集上的表现 ({model_name}) ...")
    model.eval()

    all_labels_best_model = []
    all_predictions_best_model = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            _, predicted = outputs.max(1)

            all_labels_best_model.extend(labels.cpu().numpy())
            all_predictions_best_model.extend(predicted.cpu().numpy())

    # 生成分类报告并打印
    print(f"\n{model_name} 的最佳模型在各类别上的表现：")
    print(classification_report(all_labels_best_model, all_predictions_best_model))

    # 计算并打印最终的验证准确率
    val_acc_final = (np.array(all_predictions_best_model) == np.array(all_labels_best_model)).mean() * 100
    print(f'\n{model_name} 加载最佳模型后的验证集准确率: {val_acc_final:.2f}%')


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
    model = LSTMClassifier(
        input_size=3,          # 三轴加速度数据
        hidden_size=64,        # LSTM隐藏层大小
        num_layers=2,          # LSTM层数
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
        num_epochs=300,
        device=device
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 

