import numpy as np
import scipy.io

# 从MAT文件中读取数据
def load_data(life_data_path, pulse_data_path, real_rul_path):
    lifetime_data = scipy.io.loadmat(life_data_path)['new_feature_map'].astype(float).flatten()
    shock_data = scipy.io.loadmat(pulse_data_path)['new_feature_map'].astype(float).flatten()
    real_rul = scipy.io.loadmat(real_rul_path)['max_values'].astype(float).flatten()
    # 数据归一化
    lifetime_data = (lifetime_data - np.min(lifetime_data)) / (np.max(lifetime_data) - np.min(lifetime_data))
    shock_data = (shock_data - np.min(shock_data)) / (np.max(shock_data) - np.min(shock_data))
    return lifetime_data, shock_data, real_rul

def main(data_type='c1'):
    life_data_path = f'../data/RUL_relative_similarity_{data_type}.mat'
    pulse_data_path = f'../data/Pulse_feature_{data_type}.mat'
    real_rul_path = f'../data/Line_rul_{data_type}.mat'

    life_data, shock_data, real_rul = load_data(life_data_path, pulse_data_path, real_rul_path)
    return life_data, shock_data, real_rul

if __name__ == "__main__":
    for data_type in ['c1', 'c4', 'c6']:
        life_data, shock_data, real_rul = main(data_type)
        print(f"Data {data_type.upper()}:")
        print(f"Life Data: {life_data[:5]}")
        print(f"Shock Data: {shock_data[:5]}")
        print(f"Real RUL: {real_rul[:5]}")
