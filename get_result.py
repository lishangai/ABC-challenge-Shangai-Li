import pandas as pd
import joblib
from collections import Counter

def get_utc_minute_key(timestamp_str):
    """
    将来自测试集的时间戳字符串（假定为 +0900 时区）转换为 UTC 时间的分钟级别字符串
    格式为 'YYYY-MM-DD HH:MM'
    """
    try:
        # 根据示例格式解析时间戳，例如 '2024/9/9 7:47:00'
        dt = pd.to_datetime(timestamp_str, format='ISO8601', errors='raise')
    except Exception as e:
        print(f"[WARN] 时间解析失败: {timestamp_str}, 错误: {e}")
        return None
    
    # 假设输入时间均为 Asia/Tokyo 时区（+0900）
    dt = dt.tz_localize('Asia/Tokyo')
    dt_utc = dt.tz_convert('UTC')
    minute_key = dt_utc.strftime('%Y-%m-%d %H:%M')
    return minute_key

def get_activity_label_for_minute(predictions_for_minute):
    """
    对该分钟内的所有预测活动类型进行投票，返回出现次数最多的活动类型
    """
    activity_counts = Counter([pred[0] for pred in predictions_for_minute])  # 只提取活动标签
    most_common_activity, _ = activity_counts.most_common(1)[0]
    return most_common_activity

import random

import random

def assign_activity_labels(df, predictions_by_minute):
    """
    为测试集 DataFrame 的每一行数据分配活动类型标签：
      - 如果 'Started' 和 'Finished' 均缺失，则使用 'Updated' 的时间
      - 将该时间从 +0900 转换为 UTC 的分钟级别键，并在预测结果字典中查找对应的预测数据
      - 如果找不到预测结果，则随机预测为 2806～2815 之间的一个标签，并打印提示
    """
    activity_labels = []
    total_rows = len(df)
    
    print(f"[INFO] 开始为 {total_rows} 行数据分配活动标签...")
    for index, row in df.iterrows():
        # 根据规则选择时间戳
        if pd.isna(row['Started']) and pd.isna(row['Finished']):
            ts = row['Updated']
        else:
            ts = row['Started']
        
        minute_key = get_utc_minute_key(ts)
        if not minute_key:
            print(f"[WARN] 第 {index} 行时间转换失败, 使用默认标签 'A'")
            predicted_activity = 'A'
        elif minute_key in predictions_by_minute:
            predictions_for_minute = predictions_by_minute[minute_key]
            predicted_activity = get_activity_label_for_minute(predictions_for_minute)
            # 如果预测结果为数字且在 0～9，则映射为 2806～2815
            if isinstance(predicted_activity, int) and 0 <= predicted_activity <= 9:
                predicted_activity = predicted_activity + 2806
        else:
            # 找不到预测结果时，随机选择 2806～2815 之间的标签
            predicted_activity = random.choice(range(2806, 2816))
            print(f"[WARN] 第 {index} 行对应的 UTC 时间 {minute_key} 在预测数据中未找到，随机选择标签 {predicted_activity}")
        
        activity_labels.append(predicted_activity)
        
        # 调试信息直接打印映射后的标签
        print(f"[DEBUG] 处理第 {index + 1} 行: 时间 {ts}, 转换键 {minute_key}, 预测标签 {predicted_activity}")
    
    df['PredictedActivity'] = activity_labels
    print(f"[INFO] 完成所有数据行的活动标签分配。")
    return df



def load_predictions(pkl_file):
    """
    加载存有预测结果的 pkl 文件（字典：键为 UTC 分钟字符串，值为该分钟内的预测活动列表）
    """
    try:
        predictions_by_minute = joblib.load(pkl_file)
        print(f"[INFO] 成功加载预测数据，共 {len(predictions_by_minute)} 个分钟级键。")
        return predictions_by_minute
    except Exception as e:
        print(f"[ERROR] 加载预测文件失败: {pkl_file}, 错误: {e}")
        return {}

def map_activity_labels(df):
    """
    将活动标签从 0～9 映射为 2806～2815
    如果标签是数字，则执行映射；否则保持原值不变。
    """
    def mapper(x):
        try:
            num = int(x)
            if 0 <= num <= 9:
                return num + 2806
            else:
                return x
        except:
            return x
    df['PredictedActivity'] = df['PredictedActivity'].apply(mapper)
    return df

def process_test_set(test_set_path, pkl_file, output_path):
    """
    处理测试集：
      1. 加载测试集 CSV 文件
      2. 加载预测结果数据
      3. 根据时间转换及投票逻辑为每一行分配活动类型标签
      4. 对活动标签进行映射（0～9 映射为 2806～2815）
      5. 保存结果到新的 CSV 文件
    """
    try:
        print(f"[INFO] 正在加载测试集: {test_set_path}")
        df = pd.read_csv(test_set_path)
        print(f"[INFO] 测试集加载成功，共 {len(df)} 行数据。")
    except Exception as e:
        print(f"[ERROR] 加载测试集失败: {test_set_path}, 错误: {e}")
        return

    # 加载预测数据字典
    predictions_by_minute = load_predictions(pkl_file)
    if not predictions_by_minute:
        print(f"[ERROR] 预测数据加载失败，程序终止。")
        return
    
    # 为每一行分配活动标签
    df = assign_activity_labels(df, predictions_by_minute)
    # 映射标签：0～9 转换为 2806～2815
    df = map_activity_labels(df)
    
    try:
        df.to_csv(output_path, index=False)
        print(f"[INFO] 结果已保存到 {output_path}")
    except Exception as e:
        print(f"[ERROR] 保存结果文件失败: {output_path}, 错误: {e}")

if __name__ == "__main__":
    test_set_path = 'TestActivities.csv'                  # 测试集文件路径
    pkl_file = 'predictions_by_minute.pkl'                 # 预测结果的 pkl 文件路径
    output_path = 'TestActivities_with_predictions.csv'    # 输出文件路径

    print("[INFO] 程序开始运行...")
    process_test_set(test_set_path, pkl_file, output_path)
    print("[INFO] 程序运行结束。")
