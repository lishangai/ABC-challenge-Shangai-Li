from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
import pandas as pd
import numpy as np
import time

def main():
    # 初始化数据加载器
    data_loader = DataLoader(
        base_dir='activity_segments/',
        activity_file='TrainActivities.csv'
    )
    
    # 加载数据
    print("Loading accelerometer data...")
    data = data_loader.load_data()
    
    # 特征提取
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor(window_size=64, overlap=0.5)
    segments, labels = feature_extractor.segment_data(data)
    
    print(f"\nFeature extraction:")
    # 从所有段中提取特征
    features = []
    for i, segment in enumerate(segments):
        if i % 100 == 0:  # 每处理100个片段打印一次进度
            print(f"Processing segment {i+1}/{len(segments)}")
        segment_features = feature_extractor.extract_features(segment)
        features.append(list(segment_features.values()))
    
    # 转换为numpy数组
    features = np.array(features)
    print(f"\nFeatures shape: {features.shape}")
    
    # 训练随机森林模型
    print("\nTraining Random Forest model...")
    model_trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(features, labels)
    
    # 记录训练开始时间
    start_time = time.time()
    
    model_trainer.build_model()
    model_trainer.train(X_train, y_train)
    
    # 计算训练时间
    training_time = time.time() - start_time
    print(f"\nTraining Time: {training_time:.2f} seconds")
    
    # 评估模型
    print("\nModel Evaluation Results:")
    evaluation_results = model_trainer.evaluate(X_test, y_test)
    print(evaluation_results)
    
    # 保存模型和评估结果
    print("\nSaving model and results...")
    model_trainer.save_model('saved_model.pkl', 'saved_scaler.pkl')
    
    # 保存结果到文件
    with open('model_results.txt', 'w') as f:
        f.write("Random Forest Model Results\n")
        f.write("==========================\n\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n\n")
        f.write("Evaluation Results:\n")
        f.write(evaluation_results)

if __name__ == "__main__":
    main() 