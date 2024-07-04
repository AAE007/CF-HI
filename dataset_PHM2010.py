import os
import csv
import numpy as np
import scipy.io
import pandas as pd  # 导入 pandas 以便保存 Excel
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import pywt


def get_filenames(folder_path):
    """
    获取指定文件夹下的所有文件名

    Args:
        folder_path: 文件夹路径

    Returns:
        文件名列表
    """
    filenames = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.startswith('.'):  # 过滤掉隐藏文件
                filenames.append(os.path.join(root, filename))
    return filenames


def read_csv_to_numpy(csv_file):
    """
    将 CSV 文件中的数值数据读入 NumPy 数组，并跳过非数值行

    Args:
        csv_file: CSV 文件路径

    Returns:
        NumPy 数组
    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            try:
                # 将每一行转换为浮点数，如果失败则跳过该行
                float_row = [float(item) for item in row]
                data.append(float_row)
            except ValueError:
                continue  # 跳过非数值行
    return np.array(data)


def extract_features(data):
    """
    提取时域和频域特征

    Args:
        data: 数据矩阵

    Returns:
        特征矩阵
    """
    data[np.isinf(data)] = 0
    data = np.nan_to_num(data)
    num_columns = data.shape[1]
    features_matrix = []

    for i in range(num_columns):
        col_data = data[:, i]

        # 时域特征
        mean_val = np.mean(col_data)
        rms_val = np.sqrt(np.mean(col_data ** 2))
        std_val = np.std(col_data)
        waveform_factor = rms_val / np.mean(np.abs(col_data))
        skewness_val = skew(col_data)
        kurtosis_val = kurtosis(col_data)
        peak_val = np.max(col_data)
        crest_factor = peak_val / rms_val if rms_val != 0 else 0
        impulse_factor = peak_val / np.mean(np.abs(col_data)) if np.mean(np.abs(col_data)) != 0 else 0

        # 频域特征
        fft_vals = fft(col_data)
        fft_abs = np.abs(fft_vals)
        power_spectrum = np.abs(fft_vals) ** 2
        psd_freq, psd = welch(col_data, nperseg=len(col_data) // 2)

        spectral_mean = np.mean(power_spectrum)
        spectral_centroid = np.sum(psd_freq * psd) / np.sum(psd)
        spectral_msf = np.sum((psd_freq ** 2) * psd) / np.sum(psd)

        # 小波包分解能量
        wp = pywt.WaveletPacket(data=col_data, wavelet='db4', mode='symmetric')
        wp_energy = np.sum([np.sum(node.data ** 2) for node in wp.get_level(wp.maxlevel, 'natural')])

        features = [
            mean_val, rms_val, std_val, waveform_factor, skewness_val, kurtosis_val,
            peak_val, crest_factor, impulse_factor, spectral_mean,
            spectral_centroid, spectral_msf, wp_energy
        ]

        features_matrix.append(features)

    return np.array(features_matrix).T


def main():
    # 设置文件路径
    data_folder_path = r'E:\数据\PHM-2010\c6'

    # 读取所有 CSV 文件
    filenames = get_filenames(data_folder_path)

    # 提取每个文件的特征
    all_features = []
    for file in filenames:
        data = read_csv_to_numpy(file).astype(np.float32)
        all_features.append(extract_features(data))

    # 将所有特征堆叠成一个三维矩阵
    all_features = np.stack(all_features, axis=0).reshape(len(all_features), -1)
    all_features = np.nan_to_num(all_features)

    # 保存特征到MAT文件
    features_file_path = '../data/new_feature_map_' + data_folder_path.split('\\')[-1] + r'.mat'
    scipy.io.savemat(features_file_path, {'new_feature_map': all_features})

    # 打印特征矩阵
    print(all_features)

    # 保存特征到 Excel 文件
    columns = [
        'Mean', 'RMS', 'STD', 'Waveform Factor', 'Skewness', 'Kurtosis',
        'Peak', 'Crest Factor', 'Impulse Factor', 'Spectral Mean',
        'Spectral Centroid', 'Spectral MSF', 'Wavelet Packet Energy'
    ]

    # 检查 all_features 的列数
    num_features = all_features.shape[1]
    num_files = len(filenames)

    # 计算每个文件的特征数
    features_per_file = num_features // num_files

    # 根据文件数来生成合适的列名
    all_columns = []
    for i in range(num_files):
        all_columns.extend([f'{column}_{i + 1}' for column in columns])

    # 仅截取与 all_features 的列数匹配的列名
    all_columns = all_columns[:num_features]

    df = pd.DataFrame(all_features, columns=all_columns)
    output_excel_path = '../data/extracted_features.xlsx'
    df.to_excel(output_excel_path, index=False)
    print(f"\nExtracted features have been saved to {output_excel_path}")


if __name__ == "__main__":
    main()
