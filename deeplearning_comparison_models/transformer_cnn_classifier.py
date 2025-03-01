import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import math

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ActivityDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class CNNTransformerClassifier(nn.Module):
    """结合CNN和Transformer的分类模型"""
    def __init__(self, input_size=3, d_model=64, nhead=8, num_layers=4, num_classes=10, dropout=0.2):
        super(CNNTransformerClassifier, self).__init__()
        
        # CNN层：卷积层和池化层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Transformer输入映射层
        self.input_projection = nn.Linear(64, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, x):
        # CNN 部分
        x = x.transpose(1, 2)  # 转置，使得数据形状变为 (batch_size, channels, seq_len)
        x = self.conv1(x)      # 卷积操作
        x = self.pool(x)       # 池化操作
        x = x.transpose(1, 2)  # 再次转置，使得数据形状符合 Transformer 输入要求 (batch_size, seq_len, channels)
        
        # 输入映射
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码器
        x = self.transformer_encoder(x)
        
        # 取序列的平均值进行分类
        x = x.mean(dim=1)
        
        # 分类
        x = self.classifier(x)
        return x

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda'):
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
            torch.save(model.state_dict(), 'best_transformer_model.pth')
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
    joblib.dump(scaler, 'transformer_scaler.pkl')
    
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
    model = CNNTransformerClassifier(
        input_size=3,          # 三轴加速度数据
        d_model=64,           # Transformer隐藏层大小
        nhead=8,              # 注意力头数
        num_layers=4,         # Transformer层数
        num_classes=10,       # 10种活动类型
        dropout=0.2           # dropout比率
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