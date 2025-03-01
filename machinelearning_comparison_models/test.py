from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
import pandas as pd
import numpy as np

def test_data_loading():
    print("\n=== Testing Data Loading ===")
    data_loader = DataLoader(
        base_dir='users_timeXYZ/users/',
        activity_file='TrainActivities.csv'
    )
    
    # 测试加载加速度数据
    print("\nLoading accelerometer data...")
    sensor_data = data_loader.load_accelerometer_data()
    print(f"Accelerometer data shape: {sensor_data.shape}")
    print(f"Accelerometer data columns: {sensor_data.columns.tolist()}")
    print("\nFirst few rows of accelerometer data:")
    print(sensor_data.head())
    
    # 测试加载活动标签
    print("\nLoading activity labels...")
    activity_data = data_loader.load_activity_labels()
    print(f"Activity data shape: {activity_data.shape}")
    print(f"Activity data columns: {activity_data.columns.tolist()}")
    print("\nFirst few rows of activity data:")
    print(activity_data.head())
    
    # 测试数据合并
    print("\nMerging data...")
    merged_data = data_loader.merge_data(sensor_data, activity_data)
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Merged data columns: {merged_data.columns.tolist()}")
    print("\nFirst few rows of merged data:")
    print(merged_data.head())
    
    return merged_data

def test_feature_extraction(merged_data):
    print("\n=== Testing Feature Extraction ===")
    feature_extractor = FeatureExtractor(window_size=128, overlap=0.5)
    
    print("\nSegmenting data...")
    segments, labels = feature_extractor.segment_data(merged_data)
    print(f"Number of segments: {len(segments)}")
    print(f"Number of labels: {len(labels)}")
    
    print("\nExtracting features from first segment...")
    first_segment_features = feature_extractor.extract_features(segments[0])
    print(f"Number of features: {len(first_segment_features)}")
    print("\nFeature names:")
    print(list(first_segment_features.keys()))
    
    return segments, labels

def test_model_training(segments, labels):
    print("\n=== Testing Model Training ===")
    
    # 准备特征数据
    print("\nPreparing features...")
    features = []
    for segment in segments:
        feature_extractor = FeatureExtractor()
        segment_features = feature_extractor.extract_features(segment)
        features.append(list(segment_features.values()))
    features = np.array(features)
    print(f"Features shape: {features.shape}")
    
    # 训练模型
    print("\nTraining model...")
    model_trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(features, labels)
    
    model_trainer.build_model()
    model_trainer.train(X_train, y_train)
    
    # 评估模型
    print("\nEvaluating model...")
    evaluation_results = model_trainer.evaluate(X_test, y_test)
    print("\nModel Evaluation Results:")
    print(evaluation_results)

def main():
    try:
        # 测试数据加载
        merged_data = test_data_loading()
        
        # 测试特征提取
        segments, labels = test_feature_extraction(merged_data)
        
        # 测试模型训练
        test_model_training(segments, labels)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 