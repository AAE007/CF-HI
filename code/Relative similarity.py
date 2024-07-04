import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import savgol_filter

def normalize_features(features):
    """
    将特征矩阵的每列归一化到 [0, 1] 范围

    Args:
        features: 特征矩阵，形状为 (316, 10)

    Returns:
        normalized_features: 归一化后的特征矩阵，形状为 (316, 10)
    """
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

def normalize_and_smooth_features(features, window_size=11, poly_order=2):
    """
    将特征矩阵的每列归一化到 [0, 1] 范围，并应用Savitzky-Golay滤波器进行平滑

    Args:
        features: 特征矩阵，形状为 (316, 10)
        window_size: Savitzky-Golay滤波器的窗口大小，必须是奇数
        poly_order: Savitzky-Golay滤波器的多项式阶数

    Returns:
        smoothed_features: 归一化和平滑后的特征矩阵，形状为 (316, 10)
    """
    # 归一化特征矩阵
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # 平滑特征矩阵
    smoothed_features = np.zeros_like(normalized_features)
    for i in range(normalized_features.shape[1]):
        smoothed_features[:, i] = savgol_filter(normalized_features[:, i], window_size, poly_order)

    return smoothed_features

def calculate_health_index(features, baseline):
    """
    计算设备的健康指数

    Args:
        features: 特征矩阵，形状为 (316, 10)
        baseline: 健康基线特征，形状为 (10,)

    Returns:
        health_index: 健康指数，形状为 (316,)
    """
    distances = cdist(features, baseline.reshape(1, -1), metric='euclidean')
    max_distance = np.max(distances)
    health_index = 1 - (distances.flatten() / max_distance)  # 使用归一化后的距离作为健康指数
    return health_index

def standardize_features(features):
    """
    标准化特征矩阵

    Args:
        features: 特征矩阵，形状为 (316, 10)

    Returns:
        standardized_features: 标准化后的特征矩阵，形状为 (316, 10)
    """
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    return standardized_features

def plot_health_index(health_index):
    """
    绘制健康指数的折线图

    Args:
        health_index: 健康指数，形状为 (316,)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(health_index, label='Health Index')
    plt.xlabel('Sample Index')
    plt.ylabel('Health Index')
    plt.title('Health Index over Samples')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 定义MAT文件的路径
    mat_file_path = r'..\data\new_feature_map_c4.mat'

    # 读取MAT文件
    mat_data = scipy.io.loadmat(mat_file_path)
    new_feature_map = mat_data['new_feature_map']

    # 归一化特征
    normalized_features = normalize_features(new_feature_map)
    # 归一化和平滑特征
    # normalized_features = normalize_and_smooth_features(new_feature_map)

    # 假设基线是健康设备的特征均值
    baseline = np.mean(normalized_features[0:5], axis=0)  # 仅用健康状态下的前2个样本计算基线

    # 计算健康指数
    health_index = 1 - calculate_health_index(normalized_features, baseline)

    # 保存健康指数到MAT文件
    output_file_path = '../data/RUL_relative_similarity_' + mat_file_path.split('_')[-1]
    scipy.io.savemat(output_file_path, {'new_feature_map': health_index})

    # 绘制健康指数的折线图
    plot_health_index(health_index)

if __name__ == "__main__":
    main()
