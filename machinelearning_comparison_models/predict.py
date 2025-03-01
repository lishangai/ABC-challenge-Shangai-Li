from model_trainer import ModelTrainer
from feature_extractor import FeatureExtractor
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    加载并预处理CSV文件中的加速度数据
    """
    try:
        # 读取CSV文件
        data = pd.read_csv(
            file_path,
            header=None,
            names=['RandomID', 'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
        )
        
        # 只保留加速度数据列
        return data[['Accel_X', 'Accel_Y', 'Accel_Z']].values
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def predict_activity(data, model_trainer, feature_extractor):
    """
    对输入的加速度数据进行活动类型预测
    """
    try:
        # 确保数据点足够
        if len(data) < feature_extractor.window_size:
            raise ValueError(f"数据点数量不足，需要至少{feature_extractor.window_size}个点")
        
        # 提取特征
        features = feature_extractor.extract_features(data)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # 标准化特征
        features_scaled = model_trainer.scaler.transform(features_array)
        
        # 预测
        prediction = model_trainer.model.predict(features_scaled)
        probabilities = model_trainer.model.predict_proba(features_scaled)
        
        return prediction[0], probabilities[0]
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None, None

def main():
    # 配置文件路径
    model_path = 'saved_model.pkl'
    scaler_path = 'saved_scaler.pkl'
    data_file = 'user-acc_2806_2024-09-01T22_41_11.625+0100_84033.csv'  # CSV文件路径
    
    try:
        # 加载模型
        print("加载模型...")
        model_trainer = ModelTrainer()
        model_trainer.load_model(model_path, scaler_path)
        
        # 创建特征提取器
        feature_extractor = FeatureExtractor()
        
        # 加载数据
        print("加载数据...")
        data = load_and_preprocess_data(data_file)
        if data is None:
            return
            
        print(f"加载了 {len(data)} 个数据点")
        
        # 进行预测
        print("进行预测...")
        prediction, probabilities = predict_activity(data, model_trainer, feature_extractor)
        
        if prediction is not None:
            print(f"\n预测结果:")
            print(f"预测的活动类别: {prediction}")
            print("\n各类别的概率:")
            # 显示所有可能的活动类别的概率
            activity_types = {
                2806: "Walking",
                2807: "Running",
                2808: "Stairs (ascending)",
                2809: "Stairs (descending)",
                2810: "Standing",
                2811: "Sitting",
                2812: "Lying",
                2813: "Bending",
                2814: "Picking",
                2815: "Other"
            }
            
            for activity_id, prob in zip(sorted(activity_types.keys()), probabilities):
                print(f"活动 {activity_id} ({activity_types[activity_id]}): {prob:.4f}")
            
            # 输出置信度
            confidence = max(probabilities)
            print(f"\n预测置信度: {confidence:.4f}")
            
            if confidence < 0.5:
                print("警告：预测置信度较低，结果可能不可靠")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 