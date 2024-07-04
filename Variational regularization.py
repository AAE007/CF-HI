import numpy as np
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import os
import csv
import time
import scipy.io

def get_filenames(folder_path):
    filenames = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.startswith('.'):  # 过滤掉隐藏文件
                filenames.append(os.path.join(root, filename))
    return filenames

def read_csv_to_numpy(csv_file):
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            data = [row for row in reader]
        return np.array(data, dtype=np.float16)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return np.array([])

def normalize_columns(data):
    normalized_data = []
    for matrix in data:
        if matrix.size == 0:
            continue
        normalized_matrix = np.zeros_like(matrix)
        for col in range(matrix.shape[1]):
            column = matrix[:, col]
            min_val = np.min(column)
            max_val = np.max(column)
            if max_val - min_val > 0:
                normalized_matrix[:, col] = (column - min_val) / (max_val - min_val + 1e-8)  # 加1e-8防止除以零
        normalized_data.append(normalized_matrix)
    return normalized_data

def compute_tv_regularization_values(data, weight=0.1, psi=1):
    num_elements = len(data)
    V_values = np.zeros(num_elements)

    for i in range(num_elements):
        matrix = data[i]
        if matrix.size == 0:
            continue
        K = matrix.shape[1]  # 传感器通道数量
        I = matrix.shape[0]  # 信号长度
        V_sum = 0

        for col in range(K):
            column = matrix[:, col]
            tv_column = denoise_tv_chambolle(column, weight=weight)
            delta_column = np.diff(tv_column, prepend=tv_column[0])  # 计算相邻元素之差
            ln_effect = np.log(1 + np.abs(tv_column))  # 计算对数效应

            V_sum += np.sum(delta_column + psi * ln_effect)

        V_values[i] = V_sum / (K * I)

    # 过滤 V_values 中的 NaN
    V_values = np.nan_to_num(V_values, nan=0.0)

    return V_values

def calculate_diff(tv_values):
    if len(tv_values) < 2:
        return np.zeros_like(tv_values)
    return np.diff(tv_values, axis=0)

def calculate_3sigma_limits(tv_diff):
    if tv_diff.size == 0:
        return 0, 0
    mean_diff = np.nanmean(tv_diff)
    std_diff = np.nanstd(tv_diff)
    upper_limit = mean_diff + 3 * std_diff
    lower_limit = mean_diff - 3 * std_diff
    return upper_limit, lower_limit

def plot_tv_diff_with_limits(tv_diff, upper_limit, lower_limit):
    plt.figure(figsize=(10, 6))
    plt.plot(tv_diff, label='Difference of TV Regularization Values')
    plt.axhline(y=upper_limit, color='r', linestyle='--', label='Upper 3σ limit')
    plt.axhline(y=lower_limit, color='g', linestyle='--', label='Lower 3σ limit')
    plt.xlabel('Row Index')
    plt.ylabel('Difference Value')
    plt.title('Difference of Total Variation Regularization Values for Each Row')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 设置文件路径
    data_folder_path = r'E:\数据\PHM-2010\c6'
    rul_file_path = r"C:\数据\c6\c6_wear.csv"

    # 读取所有 CSV 文件
    filenames = get_filenames(data_folder_path)

    # 计算特征提取的时间
    start_time = time.time()

    # 提取每个文件的特征
    all_features = [read_csv_to_numpy(file) for file in filenames]

    # 确保特征列表不为空
    if len(all_features) == 0 or all(len(f) == 0 for f in all_features):
        print("No valid data found.")
        return

    # 归一化数据
    all_features = normalize_columns(all_features)

    # 计算全变分正则化值和复杂化后的公式 (9) 中的信号变化率
    V_values = compute_tv_regularization_values(all_features, weight=0.1, psi=0.1)

    # 保存新的特征矩阵到MAT文件
    mat_file_path = '../data/Pulse_feature_' + data_folder_path.split('\\')[-1] + r'.mat'
    scipy.io.savemat(mat_file_path, {'new_feature_map': V_values})

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for feature extraction: {total_time:.4f} seconds")

    # 计算3σ上下限
    tv_diff = calculate_diff(V_values)
    upper_limit, lower_limit = calculate_3sigma_limits(tv_diff)

    # 绘制折线图
    plot_tv_diff_with_limits(tv_diff, upper_limit, lower_limit)

    # 打印3σ上下限
    print(f'Upper 3σ limit: {upper_limit}')
    print(f'Lower 3σ limit: {lower_limit}')

if __name__ == "__main__":
    main()
