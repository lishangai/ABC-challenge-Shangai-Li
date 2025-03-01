import os
import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, base_dir, activity_file):
        """
        初始化数据加载器
        Args:
            base_dir: 加速度计数据的根目录
            activity_file: 活动标签文件路径（用于获取有效的活动类型）
        """
        self.base_dir = base_dir
        self.activity_file = activity_file
        # 加载活动标签文件以获取有效的活动类型
        self.valid_activity_types = self._get_valid_activity_types()
        
    def _get_valid_activity_types(self):
        """获取活动标签文件中的有效活动类型"""
        try:
            labels = pd.read_csv(self.activity_file)
            valid_types = labels['Activity Type ID'].unique()
            print(f"\nValid activity types from {self.activity_file}:")
            activity_types = labels.groupby('Activity Type ID')['Activity Type'].first()
            for activity_id, activity_name in activity_types.items():
                print(f"ID {activity_id}: {activity_name}")
            return set(valid_types)
        except Exception as e:
            raise Exception(f"Error loading valid activity types: {str(e)}")
    
    def load_data(self):
        """
        加载所有加速度计数据，每个CSV文件代表一个完整的活动
        """
        all_data = []
        total_files = 0
        processed_files = 0
        
        # 首先计算有效文件夹中的总文件数
        for folder in os.listdir(self.base_dir):
            if folder.isdigit() and int(folder) in self.valid_activity_types:
                folder_path = os.path.join(self.base_dir, folder)
                if os.path.isdir(folder_path):
                    total_files += len([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        
        print(f"\nFound {total_files} CSV files in valid activity folders")
        
        # 处理每个活动类型文件夹
        for folder in os.listdir(self.base_dir):
            if not folder.isdigit():
                continue
                
            activity_type_id = int(folder)
            if activity_type_id not in self.valid_activity_types:
                continue
                
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.isdir(folder_path):
                print(f"\nProcessing activity type {activity_type_id}...")
                
                # 处理文件夹下的每个CSV文件
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(folder_path, file)
                        try:
                            # 读取CSV文件
                            data = pd.read_csv(
                                file_path,
                                header=None,
                                names=['RandomID', 'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
                            )
                            
                            # 转换时间戳
                            data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='mixed',utc=True)
                            data['Timestamp'] = data['Timestamp'].dt.tz_localize(None)
                            
                            # 确保数据按时间排序
                            data = data.sort_values('Timestamp').reset_index(drop=True)
                            
                            # 检查数据的时间连续性
                            time_diff = data['Timestamp'].diff().dropna()
                            max_gap = time_diff.max().total_seconds()
                            if max_gap > 1.0:
                                print(f"Warning: File {file} has time gaps > 1s, max gap: {max_gap:.2f}s")
                            
                            # 添加活动类型ID和文件信息
                            data['Activity Type ID'] = activity_type_id
                            data['Source_File'] = file
                            
                            all_data.append(data)
                            processed_files += 1
                            
                            # 打印进度
                            if processed_files % 10 == 0:
                                print(f"Processed {processed_files}/{total_files} files")
                            
                        except Exception as e:
                            print(f"Error processing file {file_path}: {str(e)}")
                            continue
        
        if not all_data:
            raise ValueError("No valid data found in the specified directory")
            
        # 合并所有数据
        print("\nMerging all data...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"\nData loading completed:")
        print(f"Total samples: {len(combined_data)}")
        print(f"Activity types found: {sorted(combined_data['Activity Type ID'].unique())}")
        print(f"Files per activity type:")
        print(combined_data.groupby('Activity Type ID')['Source_File'].nunique().sort_index())
        print(f"Total files processed: {processed_files}")
        
        return combined_data 