import torch
import numpy as np
import pandas as pd
from cnn_lstm_attention_classifier_all import CNNLSTMWithAttention  # 导入正确的模型类
import joblib
import os

def load_file_data(file_path, window_size=128):
    """
    加载单个CSV文件并处理成多个序列
    """
    all_sequences = []
    
    try:
        # 读取CSV文件
        data = pd.read_csv(
            file_path,
            header=None,
            names=[ 'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
         )
        #   names=['RandomID', 'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
        # )
        
        # 提取加速度数据
        sensor_data = data[['Accel_X', 'Accel_Y', 'Accel_Z']].values
        
        # 使用滑动窗口分割数据
        for i in range(0, len(sensor_data) - window_size + 1, window_size // 2):
            sequence = sensor_data[i:i + window_size]
            if len(sequence) == window_size:
                all_sequences.append(sequence)
                
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
    
    return np.array(all_sequences)

def predict_activity_type(model, sequences, scaler, device):
    """
    对每个滑动窗口进行预测并输出每个窗口的结果
    """
    model.eval()
    all_predictions = []
    
    # 对每个序列进行预测
    with torch.no_grad():
        for idx, sequence in enumerate(sequences):
            # 标准化数据（scaler处理的是二维数据，每行3个特征）
            sequence_scaled = scaler.transform(sequence)
            # 转换为tensor，形状变为 (1, window_size, 3)
            sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            # 前向传播，注意接收两个返回值：预测结果和 attention 权重
            outputs, _ = model(sequence_tensor)
            # 计算 softmax 概率分布
            probs = torch.softmax(outputs, dim=1)
            # 获取预测结果
            _, predicted = outputs.max(1)
            predicted_class = predicted.item()
            probability = probs[0][predicted_class].item()
            
            # 保存每个窗口的预测结果和概率
            all_predictions.append((idx, predicted_class, probability))
    
    return all_predictions

def group_predictions(all_predictions):
    """
    将连续相同的滑动窗口预测结果合并为一条，
    结果包含对应窗口索引列表、预测类别和平均概率
    """
    if not all_predictions:
        return []
    
    grouped = []
    current_group = [all_predictions[0]]  # 存储连续相同预测的窗口
    
    for pred in all_predictions[1:]:
        # 如果当前窗口与前一个窗口预测类别相同，则合并
        if pred[1] == current_group[-1][1]:
            current_group.append(pred)
        else:
            # 结束当前组，保存分组合并信息
            group_indices = [p[0] for p in current_group]
            avg_prob = np.mean([p[2] for p in current_group])
            grouped.append((group_indices, current_group[0][1], avg_prob))
            # 开始新的分组
            current_group = [pred]
    
    # 添加最后一组
    if current_group:
        group_indices = [p[0] for p in current_group]
        avg_prob = np.mean([p[2] for p in current_group])
        grouped.append((group_indices, current_group[0][1], avg_prob))
    
    return grouped

def main():
    # 活动类型映射
    activity_types = {
        0: "2806",
        1: "2807",
        2: "2808",
        3: "2809",
        4: "2810",
        5: "2811",
        6: "2812",
        7: "2813",
        8: "2814",
        9: "2815"
    }
    
    # 配置
    model_path = 'best_cnn_lstm_attention_model.pth'
    scaler_path = 'cnn_lstms_attention_classifier_all_scaler.pkl'
    window_size = 128  # 与训练时一致
    
    # 输入文件路径
    file_path = "2024-09-08-22-47-00-00-00 copy 2.csv"
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise ValueError(f"文件 {file_path} 不存在")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件 {model_path} 不存在")
        
        # 检查标准化器文件是否存在
        if not os.path.exists(scaler_path):
            raise ValueError(f"标准化器文件 {scaler_path} 不存在")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 加载模型
        print("Loading model...")
        model = CNNLSTMWithAttention(
            input_size=3,
            lstm_hidden_size=128,
            num_lstm_layers=2,
            num_classes=10
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        # 加载标准化器
        print("Loading scaler...")
        scaler = joblib.load(scaler_path)
        
        # 加载数据
        print("Loading and processing data...")
        sequences = load_file_data(file_path, window_size)
        
        if sequences is None or len(sequences) == 0:
            raise ValueError("没有找到有效的数据序列")
        
        print(f"Found {len(sequences)} valid sequences")
        
        # 预测每个滑动窗口
        print("Predicting...")
        all_predictions = predict_activity_type(model, sequences, scaler, device)
        
        # 输出每个滑动窗口的预测结果
        print("\n每个滑动窗口的预测结果:")
        for idx, predicted_class, probability in all_predictions:
            print(f"滑动窗口 {idx + 1}: 预测的活动类型: {activity_types[predicted_class]}, 置信度: {probability:.2%}")
        
        # 对连续相同预测的滑动窗口进行合并
        grouped_predictions = group_predictions(all_predictions)
        
        # 输出合并后的预测结果
        print("\n合并后的预测结果:")
        for group in grouped_predictions:
            indices, predicted_class, avg_prob = group
            # 如果只有一个窗口，则只显示该窗口，否则显示起始和结束窗口（索引从1开始）
            if len(indices) == 1:
                window_str = f"窗口 {indices[0] + 1}"
            else:
                window_str = f"窗口 {indices[0] + 1} - {indices[-1] + 1}"
            print(f"{window_str}: 预测的活动类型: {activity_types[predicted_class]}, 平均置信度: {avg_prob:.2%}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
