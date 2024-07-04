import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Load data from MAT files
def load_data(life_data_path, pulse_data_path, real_rul_path):
    def load_mat_data(path, key):
        return scipy.io.loadmat(path)[key].astype(float).flatten()

    life_data = load_mat_data(life_data_path, 'new_feature_map')
    pulse_data = load_mat_data(pulse_data_path, 'new_feature_map')
    real_rul = load_mat_data(real_rul_path, 'max_values')
    return life_data, pulse_data, real_rul

# Shock detection
def detect_shocks(shock_data, window_size=50, factor=1.5, sigma=3, step_size=10):
    pulse_indices, pulse_values = [], []

    for i in range(window_size, len(shock_data), step_size):
        window = shock_data[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)
        deviations = shock_data[i - window_size:i] - mean

        upper_limit = mean + factor * sigma * std
        lower_limit = mean - factor * sigma * std

        abs_deviation = np.abs(deviations)
        max_deviation_index = np.argmax(abs_deviation)
        max_deviation_value = shock_data[i] - mean

        if max_deviation_value > (upper_limit - mean) or max_deviation_value > (mean - lower_limit):
            pulse_indices.append(i - window_size + max_deviation_index)
            pulse_values.append(max_deviation_value)

    return pulse_indices, pulse_values

# Lifetime prediction
def predict_lifetime(lifetime_data, pulse_indices, pulse_values, train_ratio, decay_factor, alpha, beta, initial_threshold, threshold_factor):
    train_size = int(train_ratio * len(lifetime_data))
    max_lifetime = np.max(lifetime_data)
    predicted_rul, pulse_line = [], np.zeros_like(lifetime_data)
    failure_threshold = np.zeros_like(lifetime_data)

    cumulative_pulse_effect = 0

    for i in range(len(lifetime_data) - train_size):
        idx = i + train_size
        if idx in pulse_indices:
            cumulative_pulse_effect += pulse_values[pulse_indices.index(idx)]
        pulse_line[idx] += cumulative_pulse_effect

        decay = decay_factor ** i
        prev_avg = np.mean(predicted_rul[-2:]) if i >= 2 else max_lifetime - lifetime_data[idx] - cumulative_pulse_effect * decay
        # remaining_lifetime = alpha * prev_avg + beta * (max_lifetime - lifetime_data[idx] - cumulative_pulse_effect * decay)
        remaining_lifetime =  (max_lifetime - lifetime_data[idx] - cumulative_pulse_effect ) - cumulative_pulse_effect

        predicted_rul.append(remaining_lifetime)

        current_threshold = initial_threshold + threshold_factor * remaining_lifetime
        failure_threshold[idx] = current_threshold

    failure_threshold[:train_size] = failure_threshold[train_size] if train_size < len(failure_threshold) else initial_threshold

    for i in range(1, len(failure_threshold)):
        if failure_threshold[i] > failure_threshold[i - 1]:
            failure_threshold[i] = failure_threshold[i - 1]

    return 1 - np.array(predicted_rul), pulse_line, failure_threshold

# Calculate accuracy
def calculate_accuracy(real_rul, predicted_values, train_ratio):
    real_rul = real_rul[int(train_ratio * len(real_rul)):]
    valid_indices = np.where(real_rul == 1)[0][0] if np.any(real_rul == 1) else len(real_rul)
    real_rul, predicted_values = real_rul[:valid_indices], predicted_values[:valid_indices]
    accuracy = np.clip(1 - np.abs((real_rul - predicted_values) / real_rul), 0, 1) * 100
    return np.mean(accuracy)

def visualize_results(lifetime_data, shock_data, real_rul, predicted_rul, pulse_indices, pulse_line, failure_threshold, lifetime_accuracy, predicted_accuracy, tool_number):
    plt.figure(figsize=(12, 8))

    # Plot Shock Data with Detected Shocks
    plt.subplot(3, 1, 1)
    plt.plot(shock_data, label='Shock Data')
    plt.scatter(pulse_indices, shock_data[pulse_indices], color='red', label='Detected Shocks')
    plt.legend()
    plt.title(f'Shock Data with Detected Shocks - Tool {tool_number}')
    plt.xlabel('Time')
    plt.ylabel('Shock Value')

    # Annotate tool number
    for idx in pulse_indices:
        plt.text(idx, shock_data[idx], str(tool_number), fontsize=9, verticalalignment='bottom')

    # Plot Lifetime Data, Real RUL, and Predicted RUL
    plt.subplot(3, 1, 2)
    plt.plot(lifetime_data, label='Lifetime Data')
    plt.plot(real_rul, label='Real RUL')
    plt.plot(range(len(lifetime_data) - len(predicted_rul), len(lifetime_data)), np.array(predicted_rul), label='Predicted RUL')
    plt.legend()
    plt.title(f'Lifetime Data, Real RUL, and Predicted RUL\nLifetime Data Accuracy: {lifetime_accuracy:.2f}%, Predicted RUL Accuracy: {predicted_accuracy:.2f}%')
    plt.xlabel('Time')
    plt.ylabel('Values')

    # Annotate tool number
    plt.text(len(lifetime_data) - 1, lifetime_data[-1], str(tool_number), fontsize=9, verticalalignment='bottom')

    # Plot Pulse Line and Failure Threshold
    plt.subplot(3, 1, 3)
    plt.plot(pulse_line, label='Pulse Line', color='orange')
    plt.plot(failure_threshold, label='Failure Threshold', color='purple')
    plt.legend()
    plt.title('Pulse Prediction Line and Failure Threshold')
    plt.xlabel('Time')
    plt.ylabel('Values')

    # 保存新的特征矩阵到MAT文件
    output_dir = r"../paper/"
    file_path = os.path.join(output_dir, f'plt_Cumulative_impact_{tool_number}.svg')
    plt.savefig(file_path, format='svg')

    # Annotate tool number
    plt.text(len(pulse_line) - 1, pulse_line[-1], str(tool_number), fontsize=9, verticalalignment='bottom')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Fixed parameters
    decay_factor = 0.985
    factor = 2
    sigma = 1
    alpha = 0.4
    beta = 0.6
    initial_threshold = 0.14
    window_size = 50
    threshold_factor = 0.1

    # tool_numbers = [3, 5, 6, 7, 4, 9, 13, 15, 20104, 20106]
    tool_numbers = [7, 20106]

    for tool_number in tool_numbers:
        life_data_path = f'../data/Cutting_Tool_{tool_number}_similarity_values.mat'
        pulse_data_path = f'../data/Cutting_Tool_{tool_number}_tv_features.mat'
        real_rul_path = f'../data/Cutting_Tool_{tool_number}_real_rul.mat'

        lifetime_data, shock_data, real_rul = load_data(life_data_path, pulse_data_path, real_rul_path)

        pulse_indices, pulse_values = detect_shocks(shock_data, window_size=window_size, factor=factor, sigma=sigma)
        predicted_rul, pulse_line, failure_threshold = predict_lifetime(lifetime_data, pulse_indices, pulse_values, 0.1, decay_factor, alpha, beta, initial_threshold, threshold_factor)
        lifetime_accuracy = calculate_accuracy(real_rul, 1 - lifetime_data / np.max(lifetime_data), 0.1)
        predicted_accuracy = calculate_accuracy(real_rul, predicted_rul, 0.1)

        print(f"\nCutting Tool: {tool_number}")
        print(f"Lifetime Data Accuracy: {lifetime_accuracy:.2f}%")
        print(f"Predicted RUL Accuracy: {predicted_accuracy:.2f}%")

        visualize_results(lifetime_data, shock_data, real_rul, predicted_rul, pulse_indices, pulse_line, failure_threshold, lifetime_accuracy, predicted_accuracy, tool_number)

if __name__ == "__main__":
    main()
