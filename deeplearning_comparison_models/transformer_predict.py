import torch
import numpy as np
import pandas as pd
from transformer_classifier import TransformerClassifier
import joblib
import os

def load_file_data(file_path, window_size=64):
    """
    加载单个CSV文件并处理
    """
    all_sequences = []
    
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
                all_sequences.append(sequence)
                
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
    
    return np.array(all_sequences)

def predict_activity_type(model, sequences, scaler, device):
    """
    对多个序列进行预测并统计结果
    """
    model.eval()
    predictions = []
    
    # 对每个序列进行预测
    with torch.no_grad():
        for sequence in sequences:
            # 标准化数据
            sequence_scaled = scaler.transform(sequence)
            # 转换为tensor
            sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            # 预测
            output = model(sequence_tensor)
            # 获取预测结果
            _, predicted = output.max(1)
            predictions.append(predicted.item())
    
    # 统计预测结果
    unique_predictions, counts = np.unique(predictions, return_counts=True)
    prediction_counts = dict(zip(unique_predictions, counts))
    
    # 获取最多的预测结果
    most_common_prediction = max(prediction_counts.items(), key=lambda x: x[1])
    confidence = most_common_prediction[1] / len(predictions)
    
    return most_common_prediction[0], confidence, prediction_counts

def main():
    # 活动类型映射
    activity_types = {
        0: "Walking (2806)",
        1: "Running (2807)",
        2: "Stairs ascending (2808)",
        3: "Stairs descending (2809)",
        4: "Standing (2810)",
        5: "Sitting (2811)",
        6: "Lying (2812)",
        7: "Bending (2813)",
        8: "Picking (2814)",
        9: "Other (2815)"
    }
    
    # 配置
    model_path = 'best_transformer_model.pth'
    scaler_path = 'transformer_scaler.pkl'
    window_size = 64
    
    # 获取输入文件路径
    file_path = "user-acc_2804_2024-09-03T12_06_02.889+0100_37787.csv"
    
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
        model = TransformerClassifier()
        model.load_state_dict(torch.load(model_path))
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
        
        # 预测
        print("Predicting...")
        prediction, confidence, prediction_counts = predict_activity_type(
            model, sequences, scaler, device
        )
        
        # 输出结果
        print("\n预测结果:")
        print(f"预测的活动类型: {activity_types[prediction]}")
        print(f"预测置信度: {confidence:.2%}")
        
        print("\n各类型预测统计:")
        for pred_class, count in prediction_counts.items():
            print(f"{activity_types[pred_class]}: {count} 次 ({count/len(sequences):.2%})")
        
        if confidence < 0.5:
            print("\n警告：预测置信度较低，结果可能不可靠")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 