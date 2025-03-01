import pandas as pd
import os
import numpy as np
from datetime import datetime

def clean_filename(name):
    """清理文件名，移除或替换非法字符"""
    # 替换特殊字符
    replacements = {
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
        '(': '',
        ')': '',
        '-->' : 'to',  # 特别处理箭头
        '  ': ' ',     # 删除多余空格
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # 移除开头和结尾的空格和点
    name = name.strip('. ')
    return name

def split_activities_by_type(data, activity_labels, min_length=15):
    """
    将连续的活动数据按类别分割并保存
    
    参数:
    - data: 清洗后的传感器数据DataFrame
    - activity_labels: 活动标签数据DataFrame
    - min_length: 最小数据长度（默认15行）
    """
    # 创建基础输出目录
    base_output_dir = "activity_segments"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # 去除重复的时间戳数据
    data = data.drop_duplicates(subset=['Subject', 'Timestamp'], keep='first').copy()
    
    # 确保数据按时间和受试者排序
    data = data.sort_values(['Subject', 'Timestamp']).reset_index(drop=True)
    
    # 创建活动类型映射字典，用于目录命名
    activity_types = {}
    for _, row in activity_labels.drop_duplicates(subset=['Activity Type ID', 'Activity Type']).iterrows():
        activity_id = row['Activity Type ID']
        activity_types[activity_id] = str(activity_id)
        
        # 为每种活动类型创建目录
        activity_dir = os.path.join(base_output_dir, activity_types[activity_id])
        if not os.path.exists(activity_dir):
            os.makedirs(activity_dir)
    
    # 检测活动变化点（活动类型或受试者改变的地方）
    activity_changes = (
        (data['Activity Type ID'] != data['Activity Type ID'].shift()) | 
        (data['Subject'] != data['Subject'].shift()) |
        # 检测时间间隔过大的情况（比如超过1秒）
        (data['Timestamp'].diff().dt.total_seconds() > 1)
    )
    
    # 获取分段的索引
    segment_indices = np.where(activity_changes)[0]
    segment_indices = np.concatenate(([0], segment_indices, [len(data)]))
    
    # 用于统计每个活动类别的样本数
    activity_counts = {}
    
    # 处理每个分段
    for i in range(len(segment_indices) - 1):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i + 1]
        
        # 获取当前段的数据
        segment = data.iloc[start_idx:end_idx].copy()
        
        # 如果数据长度小于最小要求，跳过
        if len(segment) < min_length:
            continue
        
        # 获取该段的活动类型和受试者
        activity_id = segment['Activity Type ID'].iloc[0]
        subject = segment['Subject'].iloc[0]
        
        # 如果是无效的活动类型，跳过
        if activity_id not in activity_types:
            continue
            
        # 更新计数
        if activity_id not in activity_counts:
            activity_counts[activity_id] = 0
        activity_counts[activity_id] += 1
        
        # 生成文件名（包含时间戳）
        start_time = pd.to_datetime(segment['Timestamp'].iloc[0]).strftime('%Y%m%d_%H%M%S')
        end_time = pd.to_datetime(segment['Timestamp'].iloc[-1]).strftime('%Y%m%d_%H%M%S')
        filename = f"subject_{subject}_segment_{activity_counts[activity_id]}_{start_time}_to_{end_time}.csv"
        
        # 获取完整的保存路径
        filepath = os.path.join(base_output_dir, activity_types[activity_id], filename)
        
        # 只保留需要的五列并重新排序
        segment_to_save = pd.DataFrame({
            'Activity Type ID': segment['Activity Type ID'],
            'Timestamp': segment['Timestamp'],
            'Accel_X': segment['Accel_X'],
            'Accel_Y': segment['Accel_Y'],
            'Accel_Z': segment['Accel_Z']
        })
        
        # 保存数据段，不包含列名
        segment_to_save.to_csv(filepath, index=False, header=False)
        
        # 打印进度信息
        print(f"保存活动 {activity_types[activity_id]} 的第 {activity_counts[activity_id]} 个样本，"
              f"来自受试者 {subject}，数据长度: {len(segment)}")
        print(f"时间范围: {start_time} 到 {end_time}")
    
    # 打印总体统计信息
    print("\n=== 数据分割统计 ===")
    print("每个活动类别的样本数：")
    for activity_id in sorted(activity_counts.keys()):
        print(f"活动 {activity_types[activity_id]}: {activity_counts[activity_id]} 个样本")

# 使用示例
if __name__ == "__main__":
    # 读取活动标签数据
    activity_labels = pd.read_csv("TrainActivities.csv")
    
    # 读取清洗后的传感器数据
    data = pd.read_csv("cleaned_data.csv")
    
    # 确保时间戳格式正确 - 使用混合格式解析
    data['Timestamp'] = pd.to_datetime(
        data['Timestamp'], 
        format='mixed',  # 允许混合格式
        utc=True        # 确保所有时间都转换为UTC
    )
    
    # 分割并保存连续活动数据
    split_activities_by_type(data, activity_labels, min_length=15)