from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class ModelTrainer:
    def __init__(self):
        """
        初始化模型训练器
        """
        self.model = None
        self.scaler = StandardScaler()
        
    def save_model(self, model_path='saved_model.pkl', scaler_path='saved_scaler.pkl'):
        """
        保存模型和标准化器到文件
        """
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        # 保存模型
        joblib.dump(self.model, model_path)
        # 保存标准化器
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='saved_model.pkl', scaler_path='saved_scaler.pkl'):
        """
        从文件加载模型和标准化器
        """
        # 加载模型
        self.model = joblib.load(model_path)
        # 加载标准化器
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
        
    def prepare_data(self, features, labels):
        """
        准备训练数据
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """
        构建随机森林模型
        """
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """
        训练模型
        """
        # 检查每个类别的样本数量
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        min_samples = np.min(label_counts)
        
        print(f"\nClass distribution:")
        for label, count in zip(unique_labels, label_counts):
            print(f"Class {label}: {count} samples")
        
        # 动态调整交叉验证折数
        n_splits = min(3, min_samples)  # 使用最小类别样本数和3的较小值
        if n_splits < 2:
            print("\nWarning: Not enough samples for cross-validation. Proceeding with simple train/test split.")
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            print(f"\nTraining accuracy: {train_score:.3f}")
        else:
            print(f"\nUsing {n_splits}-fold cross-validation")
            self.model.fit(X_train, y_train)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=n_splits)
            print(f"\nCross-validation scores: {cv_scores}")
            print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        """
        predictions = self.model.predict(X_test)
        
        # 计算分类报告
        class_report = classification_report(y_test, predictions)
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, predictions)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # 特征重要性分析
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X_test.shape[1])],
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Most Important Features')
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return class_report 