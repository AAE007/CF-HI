#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import concurrent.futures
import random
import pandas as pd
from typing import List, Tuple, Callable, Optional

# -------------------- 全局设置 --------------------
plt.rcParams.update({'font.size': 8, 'font.family': 'Times New Roman'})
TOOL_NUMBERS = [20104, 20106]

# 参数边界：(decay_factor, factor, sigma, alpha, beta, initial_threshold, window_size, threshold_factor, train_ratio)
BOUNDS: List[Tuple[float, float]] = [
    (0.8, 1.0),
    (2.0, 2.0),
    (1.0, 1.0),
    (0.01, 0.8),
    (0.01, 1.0),
    (0.01, 1.0),
    (10, 60),
    (0.01, 1.0),
    (0.01, 0.1)
]

# -------------------- 数据加载 --------------------
def load_mat_data(path: str, key: str) -> np.ndarray:
    """加载MAT文件数据."""
    try:
        data = scipy.io.loadmat(path)[key].astype(float).flatten()
        return data
    except FileNotFoundError:
        print(f"错误: MAT 文件未找到: {path}")
    except KeyError:
        print(f"错误: 键 '{key}' 未找到: {path}")
    except Exception as e:
        print(f"加载MAT文件 {path} 失败，键为 {key}。错误信息：{e}")
    return np.array([])


def load_data(life_path: str, pulse_path: str, rul_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载寿命、脉冲与真实RUL数据."""
    life_data = load_mat_data(life_path, 'new_feature_map')
    pulse_data = load_mat_data(pulse_path, 'new_feature_map')
    real_rul = load_mat_data(rul_path, 'max_values')

    if life_data.size == 0:
        print(f"警告: 寿命数据加载失败: {life_path}")
    if pulse_data.size == 0:
        print(f"警告: 脉冲数据加载失败: {pulse_path}")
    if real_rul.size == 0:
        print(f"警告: 真实RUL数据加载失败: {rul_path}")

    return life_data, pulse_data, real_rul

# -------------------- 冲击检测 --------------------
def detect_shocks(shock_data: np.ndarray, window_size: int = 50, factor: int = 2,
                  sigma: int = 3, step_size: int = 1) -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
    """
    检测冲击事件，并计算采样点对应的上界和下界（通过线性插值扩展到全序列）。
    返回：
      - pulse_indices: 冲击点索引列表
      - pulse_values: 冲击幅值（与均值差的绝对值）
      - full_upper: 完整上界数组
      - full_lower: 完整下界数组
    """
    pulse_indices, pulse_values = [], []
    computed_indices, computed_upper, computed_lower = [], [], []
    half_window = window_size // 2

    if len(shock_data) <= 2 * half_window:
        full_upper = np.full_like(shock_data, np.nan, dtype=float)
        full_lower = np.full_like(shock_data, np.nan, dtype=float)
        return pulse_indices, pulse_values, full_upper, full_lower

    for i in range(half_window, len(shock_data) - half_window, step_size):
        window = shock_data[i - half_window: i + half_window]
        mean_val = np.mean(window)
        std_val = np.std(window)
        value = shock_data[i]
        up_lim = mean_val + factor * sigma * std_val
        lo_lim = mean_val - factor * sigma * std_val

        computed_indices.append(i)
        computed_upper.append(up_lim)
        computed_lower.append(lo_lim)

        if value > up_lim or value < lo_lim:
            pulse_indices.append(i)
            pulse_values.append(abs(value - mean_val))

    all_indices = np.arange(len(shock_data))
    if len(computed_indices) >= 2:
        full_upper = np.interp(all_indices, computed_indices, computed_upper)
        full_lower = np.interp(all_indices, computed_indices, computed_lower)
    else:
        full_upper = np.full_like(shock_data, np.nan, dtype=float)
        full_lower = np.full_like(shock_data, np.nan, dtype=float)
    return pulse_indices, pulse_values, full_upper, full_lower

# -------------------- 寿命预测 --------------------
def predict_lifetime(life_data: np.ndarray, pulse_indices: List[int], pulse_values: List[float],
                     train_ratio: float, decay_factor: float, alpha: float, beta: float,
                     initial_threshold: float, threshold_factor: float, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据冲击点信息预测剩余寿命（CF-HI模型）。
    返回预测RUL、累积冲击曲线与故障阈值。
    """
    max_life = np.max(life_data)
    predicted_rul = []
    pulse_line = np.zeros_like(life_data)
    failure_threshold = np.zeros_like(life_data)
    pulse_dict = dict(zip(pulse_indices, pulse_values))
    cumulative = 0.0
    stop_prediction = False
    previous_threshold = initial_threshold

    for i in range(len(life_data)):
        pulse_effect = pulse_dict.get(i, 0)
        cumulative += pulse_effect
        pulse_line[i] = cumulative if i == 0 else max(cumulative, pulse_line[i - 1])
        decay = decay_factor ** i
        if i >= 2 and len(predicted_rul) >= 2:
            prev_avg = np.mean(predicted_rul[-2:])
        else:
            prev_avg = max_life - life_data[i] - cumulative * decay
        pred_val = alpha * prev_avg + beta * (max_life - life_data[i] - cumulative * decay)

        current_threshold = initial_threshold + threshold_factor * pred_val
        if i > 0 and current_threshold < previous_threshold:
            current_threshold = previous_threshold
        previous_threshold = current_threshold

        failure_threshold[i] = 1 - current_threshold

        if stop_prediction or pred_val >= 1:
            pred_val = 1
        if pulse_line[i] > failure_threshold[i]:
            stop_prediction = True
            pred_val = 1

        predicted_rul.append(pred_val)

    predicted_rul = np.clip(1 - np.array(predicted_rul), 0, 1)
    return predicted_rul, pulse_line, failure_threshold

# -------------------- 辅助评估与指标 --------------------
def extract_evaluation_segment(arr: np.ndarray, train_ratio: float) -> np.ndarray:
    """
    根据 train_ratio 从数组中提取评估段（从索引 int(train_ratio * len(arr)) 开始到第一个值为1处）。
    """
    start = int(train_ratio * len(arr))
    segment = arr[start:].copy()
    end_idx = np.where(segment == 1)[0]
    end = end_idx[0] if end_idx.size > 0 else len(segment)
    return segment[:end]


def calculate_accuracy(real: np.ndarray, predicted: np.ndarray, train_ratio: float) -> float:
    """计算MAE作为预测精度."""
    real_seg = extract_evaluation_segment(real, train_ratio)
    pred_seg = predicted[:len(real_seg)]
    return np.mean(np.abs(real_seg - pred_seg))


def calculate_trend_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    根据理想趋势 y_true 与预测趋势 y_pred 计算各项指标（MAE、MSE、RMSE、单调性、趋势相关性、鲁棒性、准确性）。
    """
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    diff = np.diff(y_pred)
    pos_diff = np.sum(np.maximum(diff, 0))
    total_diff = np.sum(np.abs(diff)) + 1e-10
    monotonicity = 1 - pos_diff / total_diff
    trend_corr = np.corrcoef(y_pred, y_true)[0, 1] if len(y_true) > 1 else 0
    epsilon = 1e-10
    robustness = np.mean(np.exp(-np.abs(np.diff(y_pred) / (y_pred[:-1] + epsilon)))) if len(y_true) > 1 else 0
    accuracy = np.clip(1 - np.mean(np.abs((y_pred - y_true) / (y_true + epsilon))), 0, 1)
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Monotonicity': monotonicity,
        'Trend': trend_corr,
        'Robustness': robustness,
        'Accuracy': accuracy
    }

# -------------------- 可视化 --------------------
def visualize_results(life_data: np.ndarray, shock_data: np.ndarray, true_trend: np.ndarray,
                      predicted_rul: np.ndarray, pulse_indices: List[int], pulse_line: np.ndarray,
                      failure_threshold: np.ndarray, life_acc: float, pred_acc: float, tool_no: int,
                      shock_upper: Optional[np.ndarray] = None,
                      shock_lower: Optional[np.ndarray] = None) -> None:
    """
    绘制三个子图：
      - 图1：真实RUL与预测CF-HI；
      - 图2：冲击数据及检测点，叠加上下界；
      - 图3：累积冲击与故障阈值。
    """
    print(f"开始可视化工具 {tool_no} 的结果...")
    try:
        width_in = 88 / 25.4  # 约3.46英寸
        fig, axes = plt.subplots(3, 1, figsize=(width_in, 6), constrained_layout=True)
        ax1, ax2, ax3 = axes

        # 图1：真实趋势与预测趋势
        x_true = np.arange(len(true_trend))
        pred_range = np.arange(len(life_data) - len(predicted_rul), len(life_data))
        ax1.plot(x_true, true_trend, label='True RUL', linewidth=0.5, linestyle='--', color='black')
        ax1.plot(pred_range, predicted_rul, label='CF-HI', linewidth=0.5, linestyle='-', color='#CD1818')
        ax1.set_xlabel('Feed count/times', labelpad=-7)
        ax1.set_ylabel('RUL', labelpad=-7)
        ax1.set_xlim(0, len(true_trend) - 1)
        y_min = min(np.nanmin(true_trend), np.nanmin(predicted_rul))
        y_max = max(np.nanmax(true_trend), np.nanmax(predicted_rul))
        ax1.set_ylim(y_min, y_max)
        ax1.set_xticks([0, len(true_trend) - 1])
        ax1.set_yticks([y_min, y_max])
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        handles, labels = ax1.get_legend_handles_labels()
        ax1.text(0.94, 0.95, '(a)', transform=ax1.transAxes, fontsize=8,  va='top') # 添加子图标签 (a)
        ncol = len(handles) if len(handles) < 4 else 4
        ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),
                   ncol=ncol, frameon=True, columnspacing=1.1, borderpad=0.2,
                   handletextpad=0.1, labelspacing=0.5)

        # 图2：冲击数据与检测点及上下界
        x_shock = np.arange(len(shock_data))
        ax2.plot(x_shock, shock_data, label='Variational regularization', linewidth=0.5)
        if pulse_indices:
            ax2.scatter(pulse_indices, shock_data[pulse_indices], color='#CD1818', label='Detected impacts', s=5)
        if shock_upper is not None and shock_lower is not None:
            ax2.plot(x_shock, shock_upper, label="Upper limit", color="blue", linestyle="--", linewidth=0.5)
            ax2.plot(x_shock, shock_lower, label="Lower limit", color="green", linestyle="--", linewidth=0.5)
        ax2.set_xlabel('Feed count/times', labelpad=-7)
        ax2.set_ylabel('Impact value', labelpad=-7)
        ax2.set_xlim(0, len(shock_data) - 1)
        y_min_shock = np.nanmin(shock_data)
        y_max_shock = np.nanmax(shock_data)
        ax2.set_ylim(y_min_shock, y_max_shock)
        ax2.set_xticks([0, len(shock_data) - 1])
        ax2.set_yticks([y_min_shock - 0.15, y_max_shock + 0.05])
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.text(0.94, 0.95, '(b)', transform=ax2.transAxes, fontsize=8,  va='top') # 添加子图标签 (b)
        ax2.legend(ncol=2)


        # 图3：累积冲击与故障阈值
        x_pulse = np.arange(len(pulse_line))
        ax3.plot(x_pulse, pulse_line, label='Cumulative impact', color='#F2AF00', linewidth=0.5)
        ax3.plot(x_pulse, failure_threshold, label='Sudden failure threshold', color='#8C5CAF', linewidth=0.5)
        ax3.set_xlabel('Feed count/times', labelpad=-7)
        ax3.set_ylabel('Sudden failure', labelpad=-7)
        ax3.set_xlim(0, len(pulse_line) - 1)
        y_min_pulse = min(np.nanmin(pulse_line), np.nanmin(failure_threshold))
        y_max_pulse = max(np.nanmax(pulse_line), np.nanmax(failure_threshold))
        ax3.set_ylim(y_min_pulse, y_max_pulse)
        ax3.set_xticks([0, len(pulse_line) - 1])
        ax3.set_yticks([y_min_pulse, y_max_pulse + 0.1])
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.text(0.94, 0.95, '(c)', transform=ax3.transAxes, fontsize=8, va='top') # 添加子图标签 (c)
        ax3.legend()

        output_dir = "../paper_results/"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'Tool_{tool_no}_RUL_Prediction_Integrated_CFHI_Only.png')
        plt.savefig(file_path, format='png', dpi=600)
        print(f"图像已保存到: {file_path}")
    except Exception as e:
        print(f"可视化工具 {tool_no} 时出错: {e}")
    finally:
        plt.close(fig)
        print(f"工具 {tool_no} 的图像处理完成。")

# -------------------- 理想趋势生成辅助 --------------------
def get_perfect_trend(tool_no: int, length: int) -> np.ndarray:
    """
    根据工具编号与数据长度生成理想趋势。
    对于部分工具，前N个采样点采用线性衰减（终值不同）；其他则全程线性衰减。
    """
    settings = {
        20106: (288, 0),
        20104: (305, 0),
        12: (295, 0.3),
        9: (265, 0.2),
        15: (285, 0.1)
    }
    if tool_no in settings:
        n, end_val = settings[tool_no]
        trend = np.zeros(length)
        if length >= n:
            trend[:n] = np.linspace(1, end_val, n)
        else:
            trend = np.linspace(1, end_val, length)
    else:
        trend = np.linspace(1, 0, length)
    return trend

# -------------------- 遗传算法相关 --------------------
def objective_function(params: np.ndarray) -> float:
    """
    目标函数：对每个工具计算CF-HI模型的MAE，返回所有工具平均MAE（越小越好）。
    """
    decay_factor, factor, sigma, alpha, beta, initial_threshold, window_size, threshold_factor, train_ratio = params
    factor_int, sigma_int = int(round(factor)), int(round(sigma))
    window_size_int = int(round(window_size))

    total_mae = 0.0
    valid_tools = 0

    for tool_no in TOOL_NUMBERS:
        life_path = f'../data/Cutting_Tool_{tool_no}_similarity_values.mat'
        pulse_path = f'../data/Cutting_Tool_{tool_no}_tv_features.mat'
        rul_path = f'../data/Cutting_Tool_{tool_no}_real_rul.mat'
        life_data, shock_data, real_rul = load_data(life_path, pulse_path, rul_path)
        if life_data.size == 0 or shock_data.size == 0 or real_rul.size == 0:
            print(f"警告: 工具 {tool_no} 数据不完整，跳过。")
            continue

        # 数据反转（保证健康状态为1）
        life_data = 1 - life_data

        try:
            pulse_idx, pulse_vals, _, _ = detect_shocks(shock_data, window_size=window_size_int,
                                                        factor=factor_int, sigma=sigma_int)
            predicted, _, _ = predict_lifetime(life_data, pulse_idx, pulse_vals, train_ratio,
                                               decay_factor, alpha, beta, initial_threshold,
                                               threshold_factor, window_size_int)
            perfect_trend = get_perfect_trend(tool_no, len(life_data))
            metrics = calculate_trend_metrics(perfect_trend, predicted)
            total_mae += metrics.get('MAE', np.inf)
            valid_tools += 1
        except Exception as e:
            print(f"处理工具 {tool_no} 时出错: {e}，跳过。")
            continue

    return total_mae / valid_tools if valid_tools > 0 else float('inf')


def evaluate_population(population: np.ndarray, obj_func: Callable,
                          executor: concurrent.futures.ProcessPoolExecutor) -> np.ndarray:
    """并行评估种群中所有个体的适应度，返回顺序一致的适应度数组."""
    fitness = list(executor.map(obj_func, population))
    return np.array(fitness)


def selection(population: np.ndarray, fitness: np.ndarray, num_survivors: int) -> np.ndarray:
    """选择适应度最好的前 num_survivors 个个体."""
    indices = np.argsort(fitness)
    return population[indices[:num_survivors]]


def perform_crossover(parents: np.ndarray, offspring_count: int, crossover_rate: float) -> np.ndarray:
    """随机配对父代进行交叉，生成后代."""
    offsprings = []
    num_parents, dims = parents.shape
    while len(offsprings) < offspring_count:
        i1, i2 = random.sample(range(num_parents), 2)
        p1, p2 = parents[i1], parents[i2]
        if random.random() < crossover_rate:
            cp = random.randint(1, dims - 1)
            child1 = np.concatenate((p1[:cp], p2[cp:]))
            child2 = np.concatenate((p2[:cp], p1[cp:]))
        else:
            child1, child2 = p1.copy(), p2.copy()
        offsprings.append(child1)
        if len(offsprings) < offspring_count:
            offsprings.append(child2)
    return np.array(offsprings)


def mutate_population(population: np.ndarray, mutation_rate: float,
                      bounds: List[Tuple[float, float]]) -> np.ndarray:
    """对种群中个体随机变异，并确保基因值在边界内."""
    num_individuals, dims = population.shape
    for i in range(num_individuals):
        if random.random() < mutation_rate:
            d = random.randint(0, dims - 1)
            population[i, d] += np.random.uniform(-0.1, 0.1)
            low, high = bounds[d]
            population[i, d] = np.clip(population[i, d], low, high)
    return population


def parallel_genetic_algorithm(obj_func: Callable, bounds: List[Tuple[float, float]],
                               pop_size: int, generations: int, mutation_rate: float,
                               crossover_rate: float, executor: concurrent.futures.ProcessPoolExecutor) -> Tuple[np.ndarray, float]:
    """
    并行遗传算法：利用外部进程池优化CF-HI模型参数，使目标函数（平均MAE）最小化。
    """
    dims = len(bounds)
    population = np.random.uniform(low=[b[0] for b in bounds],
                                   high=[b[1] for b in bounds],
                                   size=(pop_size, dims))
    best_fitness = float('inf')
    best_params = None

    for gen in range(generations):
        fitness = evaluate_population(population, obj_func, executor)
        avg_fit = np.mean(fitness)
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_params = population[best_idx].copy()
        print(f"GA 迭代 {gen+1}/{generations}, 平均适应度: {avg_fit:.4f}, 当前最佳: {fitness[best_idx]:.4f}, 全局最佳: {best_fitness:.4f}")

        survivors = selection(population, fitness, pop_size // 2)
        offspring_count = pop_size - survivors.shape[0]
        offsprings = perform_crossover(survivors, offspring_count, crossover_rate)
        population = np.concatenate([survivors, offsprings], axis=0)
        population = mutate_population(population, mutation_rate, bounds)

    return best_params, best_fitness

# -------------------- 单工具数据处理 --------------------
def process_tool(tool_no: int, best_params: np.ndarray) -> dict:
    """
    处理单个工具数据，返回预测结果及冲击检测（包括上下界插值）等信息。
    """
    decay_factor, factor, sigma, alpha, beta, init_threshold, window_size, thresh_factor, train_ratio = best_params
    factor_int, sigma_int = int(round(factor)), int(round(sigma))
    window_size_int = int(round(window_size))

    life_path = f'../data/Cutting_Tool_{tool_no}_similarity_values.mat'
    pulse_path = f'../data/Cutting_Tool_{tool_no}_tv_features.mat'
    rul_path = f'../data/Cutting_Tool_{tool_no}_real_rul.mat'

    print(f"开始处理工具: {tool_no}")
    life_data, shock_data, real_rul = load_data(life_path, pulse_path, rul_path)
    if life_data.size == 0 or shock_data.size == 0 or real_rul.size == 0:
        print(f"工具 {tool_no} 数据加载失败，跳过处理。")
        return {}

    real_rul = 1 - real_rul
    life_data = 1 - life_data

    try:
        pulse_idx, pulse_vals, shock_upper, shock_lower = detect_shocks(shock_data, window_size=window_size_int,
                                                                          factor=factor_int, sigma=sigma_int)
        predicted, pulse_line, failure_threshold = predict_lifetime(life_data, pulse_idx, pulse_vals, train_ratio,
                                                                      decay_factor, alpha, beta, init_threshold,
                                                                      thresh_factor, window_size_int)
        life_acc = calculate_accuracy(real_rul, life_data / np.max(life_data), train_ratio)
        pred_acc = calculate_accuracy(real_rul, predicted, train_ratio)
        print(f"工具 {tool_no} 处理完成")
        return {
            'tool_number': tool_no,
            'life_data': life_data,
            'shock_data': shock_data,
            'real_rul': real_rul,
            'predicted_rul': predicted,
            'pulse_indices': pulse_idx,
            'pulse_line': pulse_line,
            'failure_threshold': failure_threshold,
            'life_accuracy': life_acc,
            'predicted_accuracy': pred_acc,
            'train_ratio': train_ratio,
            'shock_upper': shock_upper,
            'shock_lower': shock_lower
        }
    except Exception as e:
        print(f"处理工具 {tool_no} 过程中出错: {e}")
        return {}


# -------------------- 主函数 --------------------
def main():
    """程序入口."""
    print("程序开始运行...")

    # ---------------- GA 参数寻优 ----------------
    max_generations = 100
    population_size = 560
    mutation_rate = 0.3
    crossover_rate = 0.46

    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
        print("开始参数寻优 (并行遗传算法)...")
        best_params, best_fit = parallel_genetic_algorithm(
            objective_function, BOUNDS, population_size, max_generations, mutation_rate, crossover_rate, executor
        )
        print(f"GA 寻优完成，最佳参数: {best_params}, 最佳适应度值: {best_fit:.4f}")
        print("\n跳过局部优化 (抛光) 步骤。")

        # ---------------- 工具数据处理 ----------------
        print(f"\n待处理工具编号: {TOOL_NUMBERS}")
        tool_results = {}
        future_to_tool = {executor.submit(process_tool, tn, best_params): tn for tn in TOOL_NUMBERS}
        for fut in concurrent.futures.as_completed(future_to_tool):
            res = fut.result()
            if res:
                tool_results[res.get('tool_number')] = res
            else:
                print("警告: 某工具处理失败，跳过可视化。")
        print("所有工具数据处理完成。")

        # ---------------- 可视化与指标计算 ----------------
        combined_metrics = []

        for tool_no in TOOL_NUMBERS:
            if tool_no not in tool_results:
                print(f"警告: 工具 {tool_no} 无有效结果，跳过绘图与指标计算。")
                continue
            res = tool_results[tool_no]
            life_data = res['life_data']
            shock_data = res['shock_data']
            real_rul = res['real_rul']
            predicted_rul = res['predicted_rul']
            pulse_indices = res['pulse_indices']
            pulse_line = res['pulse_line']
            failure_threshold = res['failure_threshold']
            life_acc = res['life_accuracy']
            pred_acc = res['predicted_accuracy']
            shock_upper = res.get('shock_upper')
            shock_lower = res.get('shock_lower')

            perfect_trend = get_perfect_trend(tool_no, len(life_data))

            visualize_results(
                life_data, shock_data, perfect_trend, predicted_rul, pulse_indices, pulse_line,
                failure_threshold, life_acc, pred_acc, tool_no,
                shock_upper=shock_upper, shock_lower=shock_lower
            )
            print(f"工具 {tool_no} 图像生成完成。")

            cfhi_metrics = calculate_trend_metrics(perfect_trend, predicted_rul)
            cfhi_metrics['Tool Number'] = tool_no
            cfhi_metrics['Model'] = 'CF-HI'
            combined_metrics.append(cfhi_metrics)


        output_excel = os.path.join("../paper_results", "combined_metrics_cfhi_only.xlsx")
        metrics_df = pd.DataFrame(combined_metrics)
        with pd.ExcelWriter(output_excel) as writer:
            # 1) 先将指标列从宽表格转换为长表格（melt）
            metrics_cols = ['MAE', 'MSE', 'Monotonicity', 'RMSE', 'Robustness', 'Trend', 'Accuracy']
            melted = pd.melt(
                metrics_df,
                id_vars=['Tool Number', 'Model'],  # 不动的列
                value_vars=metrics_cols,  # 要展开的指标列
                var_name='Metric',  # 长表格中，展开后的列名
                value_name='Value'  # 指标取值列名
            )

            # 2) 将长表格透视成行索引 = [Metric, Model]，列索引 = Tool Number
            pivoted_metrics = melted.pivot_table(
                index=['Metric', 'Model'],  # 多级行索引
                columns='Tool Number',  # 列为工具编号
                values='Value',  # 单元格填充数值
                aggfunc='mean'  # 避免分组冲突，一般用 mean 或 first 均可
            )

            # 3) 按指定顺序重排工具编号列（如表格示例顺序：12, 9, 15, 5, 7, 13）
            # tool_order = [20104, 20106] # You can adjust this order if needed, but for CF-HI only, it might not be necessary.
            pivoted_metrics = pivoted_metrics.reindex(columns=TOOL_NUMBERS) # Use TOOL_NUMBERS for current tool list

            # 4) 若需要特定行顺序，可手动构造 MultiIndex
            #    下面示例仅保留 CF-HI 模型的指标
            metric_order = ['MAE', 'MSE', 'Monotonicity', 'RMSE', 'Robustness', 'Trend', 'Accuracy']
            model_order = ['CF-HI'] # Only CF-HI model
            new_index = pd.MultiIndex.from_product([metric_order, model_order], names=['Metric', 'Model'])
            pivoted_metrics = pivoted_metrics.reindex(new_index)

            # 5) 写出到 Excel 的 "Metrics" sheet
            pivoted_metrics.to_excel(writer, sheet_name='Metrics')
        print(f"\n所有图像与指标已保存至 {output_excel}")
        print("程序运行结束.")

if __name__ == "__main__":
    main()