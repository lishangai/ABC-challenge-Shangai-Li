import numpy as np
from scipy import stats
from scipy.fft import fft

class FeatureExtractor:
    def __init__(self, window_size=64, overlap=0.5):
        """
        初始化特征提取器
        Args:
            window_size: 窗口大小（默认64个样本，约1秒的数据）
            overlap: 重叠比例（默认0.5，相邻窗口重叠一半）
        """
        self.window_size = window_size
        self.overlap = overlap
        
    def segment_data(self, data):
        """
        将数据分割成样本，每个文件作为一个独立的样本
        """
        segments = []
        labels = []
        
        # 按文件分组处理
        for (source_file, activity_id), group in data.groupby(['Source_File', 'Activity Type ID']):
            # 确保数据按时间排序
            group = group.sort_values('Timestamp')
            
            # 如果数据量太少，跳过
            if len(group) < self.window_size:
                print(f"Warning: File {source_file} (Activity {activity_id}) has only {len(group)} samples, skipping...")
                continue
            
            # 检查时间连续性
            time_diff = group['Timestamp'].diff().dropna()
            if time_diff.max().total_seconds() > 1.0:  # 如果样本间隔超过1秒，输出警告
                print(f"Warning: File {source_file} has time gaps > 1s, max gap: {time_diff.max().total_seconds():.2f}s")
            
            # 从文件中提取多个窗口
            n_windows = (len(group) - self.window_size) // (self.window_size // 2) + 1
            if n_windows > 0:
                for i in range(n_windows):
                    start_idx = i * (self.window_size // 2)
                    segment = group.iloc[start_idx:start_idx + self.window_size]
                    if len(segment) == self.window_size:  # 确保窗口大小正确
                        segments.append(segment[['Accel_X', 'Accel_Y', 'Accel_Z']].values)
                        labels.append(activity_id)
        
        if len(segments) == 0:
            raise ValueError("No valid segments found. Check if the data contains enough samples.")
            
        print(f"\nSegmentation summary:")
        print(f"Total segments: {len(segments)}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Activity {label}: {count} segments")
            
        return np.array(segments), np.array(labels)
    
    def extract_features(self, segment):
        """
        从一个数据段中提取特征
        """
        features = {}
        
        # 时域特征
        features.update(self._time_domain_features(segment))
        
        # 频域特征
        features.update(self._frequency_domain_features(segment))
        
        # 统计特征
        features.update(self._statistical_features(segment))
        
        return features
    
    def _time_domain_features(self, segment):
        """提取时域特征"""
        features = {}
        
        # 对每个轴计算特征
        for i, axis in enumerate(['X', 'Y', 'Z']):
            features.update({
                f'mean_{axis}': np.mean(segment[:, i]),
                f'std_{axis}': np.std(segment[:, i]),
                f'max_{axis}': np.max(segment[:, i]),
                f'min_{axis}': np.min(segment[:, i]),
                f'rms_{axis}': np.sqrt(np.mean(np.square(segment[:, i]))),
            })
            
        return features
    
    def _frequency_domain_features(self, segment):
        """提取频域特征"""
        features = {}
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            # 计算FFT
            fft_values = fft(segment[:, i])
            fft_freq = np.abs(fft_values)[:self.window_size//2]
            
            features.update({
                f'fft_mean_{axis}': np.mean(fft_freq),
                f'fft_std_{axis}': np.std(fft_freq),
                f'fft_max_{axis}': np.max(fft_freq),
            })
            
        return features
    
    def _statistical_features(self, segment):
        """提取统计特征"""
        features = {}
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            features.update({
                f'skewness_{axis}': stats.skew(segment[:, i]),
                f'kurtosis_{axis}': stats.kurtosis(segment[:, i]),
                f'zcr_{axis}': np.sum(np.diff(np.signbit(segment[:, i]))) / len(segment),
            })
            
        return features 