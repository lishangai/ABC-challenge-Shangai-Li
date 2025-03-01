import os
import torch
import pandas as pd
import numpy as np
from cnn_lstm_attention_classifier_all import CNNLSTMWithAttention
import joblib

def load_file_data(file_path, window_size=128):
    """
    加载单个CSV文件并处理成多个序列，同时返回每个序列对应的起始时间（精确到分钟）
    """
    print(f"[DEBUG] 开始处理文件: {file_path}")
    all_sequences = []
    
    try:
        # 读取CSV文件
        data = pd.read_csv(
            file_path,
            header=None,
            names=['Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
        )
        print(f"[DEBUG] 读取文件成功，数据行数: {len(data)}")
        
        
        # 提取加速度数据
        sensor_data = data[['Accel_X', 'Accel_Y', 'Accel_Z']].values
        print(f"[DEBUG] 提取加速度数据成功, shape: {sensor_data.shape}")
        
        # 使用滑动窗口分割数据，每次步长为 window_size // 2
        for i in range(0, len(sensor_data) - window_size + 1, window_size // 2):
            sequence = sensor_data[i:i + window_size]
            if len(sequence) == window_size:
                all_sequences.append(sequence)
        
        print(f"[DEBUG] 分割成 {len(all_sequences)} 个序列")
                
    except Exception as e:
        print(f"[ERROR] 处理文件 {file_path} 时发生错误: {str(e)}")
        return None
    
    return np.array(all_sequences)

def predict_activity_type(model, sequences, scaler, device):
    """
    对每个滑动窗口进行预测并输出每个窗口的结果
    返回列表，每个元素为 (window_index, predicted_class, probability)
    """
    print(f"[DEBUG] 开始对 {len(sequences)} 个序列进行预测")
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for idx, sequence in enumerate(sequences):
            # 标准化数据（scaler处理的是二维数据，每行3个特征）
            sequence_scaled = scaler.transform(sequence)
            # 转换为tensor，形状为 (1, window_size, 3)
            sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            # 前向传播（注意模型返回预测结果和attention权重）
            outputs, _ = model(sequence_tensor)
            # 计算 softmax 概率分布
            probs = torch.softmax(outputs, dim=1)
            # 获取预测结果
            _, predicted = outputs.max(1)
            predicted_class = predicted.item()
            probability = probs[0][predicted_class].item()
            
            all_predictions.append((idx, predicted_class, probability))
            print(f"[DEBUG] 序列 {idx}：预测类别 = {predicted_class}, 概率 = {probability:.4f}")
    
    print(f"[DEBUG] 预测完成，共处理 {len(all_predictions)} 个窗口")
    return all_predictions


def group_predictions_by_minute(all_predictions, timestamp_str):
    """
    根据给定的文件名时间戳字符串将每个窗口的预测结果分组存储到字典中。
    文件名格式示例: "2024-09-12-19-37-00-00-00"
    这里我们只取前5部分（YYYY-MM-DD-HH-MM），转换成 "YYYY-MM-DD HH:MM" 的格式。
    """
    predictions_by_minute = {}

    try:
        # 按 '-' 分割字符串，得到各部分
        parts = timestamp_str.split('-')
        if len(parts) < 5:
            raise ValueError("文件名格式不正确，无法提取足够的日期时间信息。")
        
        # 提取前5部分：年、月、日、小时、分钟，并组装成 "YYYY-MM-DD HH:MM" 格式
        new_timestamp = f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}"
        
        # 使用 pd.to_datetime 做进一步转换，确保格式一致
        new_timestamp = pd.to_datetime(new_timestamp, format='%Y-%m-%d %H:%M').strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        print(f"[ERROR] 时间戳格式转换失败: {timestamp_str}, 错误: {e}")
        return {}

    # 将所有预测结果（忽略窗口索引）分组到这个时间戳下
    predictions_by_minute[new_timestamp] = [(pred[1], pred[2]) for pred in all_predictions]
    
    print(f"[DEBUG] 为时间戳 {new_timestamp} 分组预测结果，共 {len(predictions_by_minute[new_timestamp])} 个窗口预测")
    return predictions_by_minute


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
    window_size = 128  # 与训练时保持一致
    
    # 文件夹路径
    folder_path = "minute_data"
    
    try:
        # 文件夹检查
        if not os.path.exists(folder_path):
            raise ValueError(f"文件夹 {folder_path} 不存在")
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件 {model_path} 不存在")
        if not os.path.exists(scaler_path):
            raise ValueError(f"标准化器文件 {scaler_path} 不存在")
        
        # 选择设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DEBUG] 使用设备: {device}")
        
        # 加载模型
        print("[DEBUG] 正在加载模型...")
        model = CNNLSTMWithAttention(
            input_size=3,
            lstm_hidden_size=128,
            num_lstm_layers=2,
            num_classes=10
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print("[DEBUG] 模型加载成功")
        
        # 加载标准化器
        print("[DEBUG] 正在加载标准化器...")
        scaler = joblib.load(scaler_path)
        print("[DEBUG] 标准化器加载成功")
        
        # 初始化结果字典
        all_predictions_by_minute = {}
        
        # 遍历文件夹中的所有CSV文件
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                
                # 提取文件名中的时间戳
                timestamp_str = filename.split('.')[0]
                print(f"[DEBUG] 开始处理文件: {filename}, 时间戳: {timestamp_str}")
                
                # 加载数据并进行预测
                sequences = load_file_data(file_path, window_size)
                if sequences is None or len(sequences) == 0:
                    print(f"[WARNING] 文件 {filename} 无有效数据，跳过")
                    continue
                else:
                    print(f"[DEBUG] 文件 {filename} 成功加载 {sequences.shape[0]} 个序列")
                
                # 预测每个滑动窗口
                all_predictions = predict_activity_type(model, sequences, scaler, device)
                
                # 根据分钟分组存储预测结果
                predictions_by_minute = group_predictions_by_minute(all_predictions, timestamp_str)
                
                # 合并到总字典中
                all_predictions_by_minute.update(predictions_by_minute)
        
        # 保存预测结果
        output_file = "predictions_by_minute.pkl"
        joblib.dump(all_predictions_by_minute, output_file)
        print(f"[DEBUG] 预测结果已保存到 {output_file}")
    
    except Exception as e:
        print(f"[ERROR] 发生错误: {str(e)}")

if __name__ == "__main__":
    main()
