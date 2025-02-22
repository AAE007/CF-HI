#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refactored Code for CF-HI Model Optimization, Lifetime Prediction, and Visualization

This script loads MAT file data, performs shock detection, predicts remaining tool life (CF-HI model),
and uses a parallel genetic algorithm to optimize model parameters. It then visualizes the results
and saves performance metrics to an Excel file.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter
import concurrent.futures
import random
import pandas as pd
from typing import List, Tuple, Callable, Optional

# -------------------- Global Settings --------------------
plt.rcParams.update({'font.size': 8, 'font.family': 'Times New Roman'})
TOOL_NUMBERS = [20104, 20106]

# Parameter bounds: (decay_factor, factor, sigma, alpha, beta, initial_threshold, window_size, threshold_factor, train_ratio)
BOUNDS: List[Tuple[float, float]] = [
    (0.8, 1.0),  # decay_factor
    (2.0, 2.0),  # factor
    (1.0, 1.0),  # sigma
    (0.01, 0.8),  # alpha
    (0.01, 1.0),  # beta
    (0.01, 1.0),  # initial_threshold
    (10, 60),  # window_size
    (0.01, 1.0),  # threshold_factor
    (0.01, 0.1)  # train_ratio
]


# -------------------- Data Loading Functions --------------------
def load_mat_data(path: str, key: str) -> np.ndarray:
    """
    Load data from a MAT file.

    Parameters:
        path (str): The path to the MAT file.
        key (str): The key used to extract data.

    Returns:
        np.ndarray: Flattened array of the data; empty array on failure.
    """
    try:
        data = scipy.io.loadmat(path)[key].astype(float).flatten()
        return data
    except FileNotFoundError:
        print(f"Error: MAT file not found: {path}")
    except KeyError:
        print(f"Error: Key '{key}' not found in: {path}")
    except Exception as e:
        print(f"Failed to load MAT file {path} with key '{key}'. Error: {e}")
    return np.array([])


def load_data(life_path: str, pulse_path: str, rul_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load life, pulse, and real RUL data from MAT files.

    Parameters:
        life_path (str): File path for life data.
        pulse_path (str): File path for pulse (shock) data.
        rul_path (str): File path for real RUL data.

    Returns:
        Tuple containing life data, pulse data, and real RUL data as numpy arrays.
    """
    life_data = load_mat_data(life_path, 'new_feature_map')
    pulse_data = load_mat_data(pulse_path, 'new_feature_map')
    real_rul = load_mat_data(rul_path, 'max_values')

    if life_data.size == 0:
        print(f"Warning: Life data failed to load from: {life_path}")
    if pulse_data.size == 0:
        print(f"Warning: Pulse data failed to load from: {pulse_path}")
    if real_rul.size == 0:
        print(f"Warning: Real RUL data failed to load from: {rul_path}")

    return life_data, pulse_data, real_rul


# -------------------- Shock Detection Function --------------------
def detect_shocks(shock_data: np.ndarray, window_size: int = 50, factor: int = 2,
                  sigma: int = 3, step_size: int = 1) -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
    """
    Detect shock events in the input data and compute the upper and lower bounds via linear interpolation.

    Parameters:
        shock_data (np.ndarray): Array of shock data.
        window_size (int): Size of the moving window.
        factor (int): Multiplicative factor for threshold.
        sigma (int): Number of standard deviations.
        step_size (int): Step size for moving window.

    Returns:
        Tuple:
            - List of indices where shocks are detected.
            - List of shock amplitudes (absolute difference from window mean).
            - Full upper bound array.
            - Full lower bound array.
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


# -------------------- Lifetime Prediction Function --------------------
def predict_lifetime(life_data: np.ndarray, pulse_indices: List[int], pulse_values: List[float],
                     train_ratio: float, decay_factor: float, alpha: float, beta: float,
                     initial_threshold: float, threshold_factor: float, window_size: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict the remaining lifetime (RUL) using the CF-HI model based on shock event data.

    Parameters:
        life_data (np.ndarray): Array representing life data.
        pulse_indices (List[int]): Indices where shocks are detected.
        pulse_values (List[float]): Shock amplitudes.
        train_ratio (float): Ratio for training/evaluation segmentation.
        decay_factor (float): Decay factor for cumulative impact.
        alpha (float): Weight parameter for previous average.
        beta (float): Weight parameter for current degradation.
        initial_threshold (float): Initial threshold value.
        threshold_factor (float): Factor to adjust threshold.
        window_size (int): Window size used in shock detection.

    Returns:
        Tuple:
            - Predicted RUL as a numpy array.
            - Cumulative pulse (impact) line.
            - Failure threshold array.
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

        # Compute previous average prediction or use default estimation
        if i >= 2 and len(predicted_rul) >= 2:
            prev_avg = np.mean(predicted_rul[-2:])
        else:
            prev_avg = max_life - life_data[i] - cumulative * decay

        pred_val = alpha * prev_avg + beta * (max_life - life_data[i] - cumulative * decay)
        current_threshold = initial_threshold + threshold_factor * pred_val

        # Ensure threshold does not decrease
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

    # Clip predicted RUL values between 0 and 1
    predicted_rul = np.clip(1 - np.array(predicted_rul), 0, 1)
    return predicted_rul, pulse_line, failure_threshold


# -------------------- Evaluation Functions --------------------
def extract_evaluation_segment(arr: np.ndarray, train_ratio: float) -> np.ndarray:
    """
    Extract evaluation segment from an array based on the training ratio.

    Parameters:
        arr (np.ndarray): Input array.
        train_ratio (float): Ratio to determine start index.

    Returns:
        np.ndarray: Segment from the training ratio index to the first occurrence of 1.
    """
    start = int(train_ratio * len(arr))
    segment = arr[start:].copy()
    end_idx = np.where(segment == 1)[0]
    end = end_idx[0] if end_idx.size > 0 else len(segment)
    return segment[:end]


def calculate_accuracy(real: np.ndarray, predicted: np.ndarray, train_ratio: float) -> float:
    """
    Calculate the Mean Absolute Error (MAE) as the prediction accuracy metric.

    Parameters:
        real (np.ndarray): Ground truth values.
        predicted (np.ndarray): Predicted values.
        train_ratio (float): Training ratio to extract evaluation segment.

    Returns:
        float: MAE value.
    """
    real_seg = extract_evaluation_segment(real, train_ratio)
    pred_seg = predicted[:len(real_seg)]
    return np.mean(np.abs(real_seg - pred_seg))


def calculate_trend_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute various trend metrics between the true and predicted trends.

    Parameters:
        y_true (np.ndarray): Ideal trend values.
        y_pred (np.ndarray): Predicted trend values.

    Returns:
        dict: Dictionary containing MAE, MSE, RMSE, Monotonicity, Trend correlation, Robustness, and Accuracy.
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


# -------------------- Visualization Function --------------------
def visualize_results(life_data: np.ndarray, shock_data: np.ndarray, true_trend: np.ndarray,
                      predicted_rul: np.ndarray, pulse_indices: List[int], pulse_line: np.ndarray,
                      failure_threshold: np.ndarray, life_acc: float, pred_acc: float, tool_no: int,
                      additional_trends: Optional[List[np.ndarray]] = None,
                      additional_trend_labels: Optional[List[str]] = None,
                      shock_upper: Optional[np.ndarray] = None,
                      shock_lower: Optional[np.ndarray] = None) -> None:
    """
    Visualize the results in three subplots:
      - Subplot (a): True RUL, predicted CF-HI, and additional trends.
      - Subplot (b): Shock data with detected points and upper/lower bounds.
      - Subplot (c): Cumulative shock (impact) and failure threshold.

    Parameters:
        life_data (np.ndarray): Life data array.
        shock_data (np.ndarray): Shock data array.
        true_trend (np.ndarray): Ideal trend data.
        predicted_rul (np.ndarray): Predicted RUL values.
        pulse_indices (List[int]): Indices of detected shock events.
        pulse_line (np.ndarray): Cumulative shock values.
        failure_threshold (np.ndarray): Failure threshold array.
        life_acc (float): Life accuracy metric.
        pred_acc (float): Prediction accuracy metric.
        tool_no (int): Tool number identifier.
        additional_trends (Optional[List[np.ndarray]]): List of additional trend arrays.
        additional_trend_labels (Optional[List[str]]): Labels for the additional trends.
        shock_upper (Optional[np.ndarray]): Upper bound for shock data.
        shock_lower (Optional[np.ndarray]): Lower bound for shock data.
    """
    print(f"Visualizing results for Tool {tool_no}...")
    try:
        width_in = 88 / 25.4  # Approximately 3.46 inches
        fig, axes = plt.subplots(3, 1, figsize=(width_in, 6), constrained_layout=True)
        ax1, ax2, ax3 = axes

        # Subplot (a): True Trend and Predicted RUL
        x_true = np.arange(len(true_trend))
        pred_range = np.arange(len(life_data) - len(predicted_rul), len(life_data))
        ax1.plot(x_true, true_trend, label='True RUL', linewidth=0.5, linestyle='--', color='black')
        if additional_trends is not None and additional_trend_labels is not None:
            colors = ['#0E4D92', '#FF7034', '#929292', "#F2AF00", '#8C5CAF', '#006400']
            linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]
            markers = ['o', 's', 'D', '^', 'v', 'p']
            x_full = np.arange(len(life_data))
            markevery = max(1, len(x_full) // 30)
            for i, (trend, label) in enumerate(zip(additional_trends, additional_trend_labels)):
                ax1.plot(x_full, trend, label=label, linewidth=0.5, color=colors[i % len(colors)],
                         linestyle=linestyles[i % len(linestyles)], marker=markers[i % len(markers)],
                         markersize=2, markevery=markevery)
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
        # Add subplot label (a)
        ax1.text(0.94, 0.95, '(a)', transform=ax1.transAxes, fontsize=8, va='top')
        handles, labels = ax1.get_legend_handles_labels()
        ncol = len(handles) if len(handles) < 4 else 4
        ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),
                   ncol=ncol, frameon=True, columnspacing=1.1, borderpad=0.2,
                   handletextpad=0.1, labelspacing=0.5)

        # Subplot (b): Shock Data with Detected Impacts and Bounds
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
        # Add subplot label (b)
        ax2.text(0.94, 0.95, '(b)', transform=ax2.transAxes, fontsize=8, va='top')
        ax2.legend(ncol=2)

        # Subplot (c): Cumulative Impact and Failure Threshold
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
        # Add subplot label (c)
        ax3.text(0.94, 0.95, '(c)', transform=ax3.transAxes, fontsize=8, va='top')
        ax3.legend()

        # Save the figure to file
        output_dir = "../paper_results/"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'Tool_{tool_no}_RUL_Prediction_Integrated.png')
        plt.savefig(file_path, format='png', dpi=600)
        print(f"Figure saved to: {file_path}")
    except Exception as e:
        print(f"Error visualizing Tool {tool_no}: {e}")
    finally:
        plt.close(fig)
        print(f"Visualization for Tool {tool_no} completed.")


# -------------------- Ideal Trend Generation --------------------
def get_perfect_trend(tool_no: int, length: int) -> np.ndarray:
    """
    Generate the ideal trend based on the tool number and data length.

    For certain tools, a linear decay is applied over the first N samples (with a specified end value);
    for others, a full linear decay is used.

    Parameters:
        tool_no (int): Tool number identifier.
        length (int): Total number of samples.

    Returns:
        np.ndarray: Ideal trend array.
    """
    settings = {
        20106: (288, 0),
        20104: (305, 0),
    }
    if tool_no in settings:
        n, end_val = settings[tool_no]
        if length >= n:
            trend = np.linspace(1, end_val, n)
            # Extend trend to full length by repeating final value
            trend = np.concatenate([trend, np.full(length - n, end_val)])
        else:
            trend = np.linspace(1, end_val, length)
    else:
        trend = np.linspace(1, 0, length)
    return trend


# -------------------- Genetic Algorithm Functions --------------------
def objective_function(params: np.ndarray) -> float:
    """
    Objective function for the genetic algorithm.
    For each tool, compute the MAE of the CF-HI model predictions and return the average MAE.

    Parameters:
        params (np.ndarray): Array of model parameters.

    Returns:
        float: Average MAE across tools (lower is better).
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
            print(f"Warning: Incomplete data for Tool {tool_no}, skipping.")
            continue

        # Invert data to ensure healthy state is 1
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
            print(f"Error processing Tool {tool_no}: {e}, skipping.")
            continue

    return total_mae / valid_tools if valid_tools > 0 else float('inf')


def evaluate_population(population: np.ndarray, obj_func: Callable,
                        executor: concurrent.futures.ProcessPoolExecutor) -> np.ndarray:
    """
    Evaluate the fitness of all individuals in the population in parallel.

    Parameters:
        population (np.ndarray): Population array.
        obj_func (Callable): Objective function to evaluate fitness.
        executor (ProcessPoolExecutor): Executor for parallel processing.

    Returns:
        np.ndarray: Array of fitness values.
    """
    fitness = list(executor.map(obj_func, population))
    return np.array(fitness)


def selection(population: np.ndarray, fitness: np.ndarray, num_survivors: int) -> np.ndarray:
    """
    Select the top-performing individuals based on fitness.

    Parameters:
        population (np.ndarray): Population array.
        fitness (np.ndarray): Fitness values for the population.
        num_survivors (int): Number of survivors to select.

    Returns:
        np.ndarray: Selected survivors.
    """
    indices = np.argsort(fitness)
    return population[indices[:num_survivors]]


def perform_crossover(parents: np.ndarray, offspring_count: int, crossover_rate: float) -> np.ndarray:
    """
    Perform crossover on parent individuals to generate offspring.

    Parameters:
        parents (np.ndarray): Array of selected parent individuals.
        offspring_count (int): Number of offspring to generate.
        crossover_rate (float): Probability of crossover.

    Returns:
        np.ndarray: Array of offspring individuals.
    """
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
    """
    Mutate individuals in the population randomly, ensuring gene values remain within specified bounds.

    Parameters:
        population (np.ndarray): Population array.
        mutation_rate (float): Mutation probability.
        bounds (List[Tuple[float, float]]): List of (min, max) bounds for each gene.

    Returns:
        np.ndarray: Mutated population.
    """
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
                               crossover_rate: float, executor: concurrent.futures.ProcessPoolExecutor) -> Tuple[
    np.ndarray, float]:
    """
    Parallel Genetic Algorithm to optimize CF-HI model parameters by minimizing the average MAE.

    Parameters:
        obj_func (Callable): Objective function to minimize.
        bounds (List[Tuple[float, float]]): Bounds for each parameter.
        pop_size (int): Population size.
        generations (int): Number of generations.
        mutation_rate (float): Mutation rate.
        crossover_rate (float): Crossover rate.
        executor (ProcessPoolExecutor): Executor for parallel processing.

    Returns:
        Tuple: Best parameters found and the corresponding best fitness value.
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
        print(
            f"GA Iteration {gen + 1}/{generations}, Avg Fitness: {avg_fit:.4f}, Current Best: {fitness[best_idx]:.4f}, Global Best: {best_fitness:.4f}")

        survivors = selection(population, fitness, pop_size // 2)
        offspring_count = pop_size - survivors.shape[0]
        offsprings = perform_crossover(survivors, offspring_count, crossover_rate)
        population = np.concatenate([survivors, offsprings], axis=0)
        population = mutate_population(population, mutation_rate, bounds)

    return best_params, best_fitness


# -------------------- Single Tool Data Processing --------------------
def process_tool(tool_no: int, best_params: np.ndarray) -> dict:
    """
    Process data for a single tool and return prediction results along with shock detection information.

    Parameters:
        tool_no (int): Tool number identifier.
        best_params (np.ndarray): Optimized model parameters.

    Returns:
        dict: Dictionary containing processed results and metrics.
    """
    decay_factor, factor, sigma, alpha, beta, init_threshold, window_size, thresh_factor, train_ratio = best_params
    factor_int, sigma_int = int(round(factor)), int(round(sigma))
    window_size_int = int(round(window_size))

    life_path = f'../data/Cutting_Tool_{tool_no}_similarity_values.mat'
    pulse_path = f'../data/Cutting_Tool_{tool_no}_tv_features.mat'
    rul_path = f'../data/Cutting_Tool_{tool_no}_real_rul.mat'

    print(f"Processing Tool {tool_no}...")
    life_data, shock_data, real_rul = load_data(life_path, pulse_path, rul_path)
    if life_data.size == 0 or shock_data.size == 0 or real_rul.size == 0:
        print(f"Tool {tool_no} data incomplete, skipping.")
        return {}

    # Invert data to ensure healthy state is represented by 1
    real_rul = 1 - real_rul
    life_data = 1 - life_data

    try:
        pulse_idx, pulse_vals, shock_upper, shock_lower = detect_shocks(
            shock_data, window_size=window_size_int, factor=factor_int, sigma=sigma_int
        )
        predicted, pulse_line, failure_threshold = predict_lifetime(
            life_data, pulse_idx, pulse_vals, train_ratio,
            decay_factor, alpha, beta, init_threshold, thresh_factor, window_size_int
        )
        life_acc = calculate_accuracy(real_rul, life_data / np.max(life_data), train_ratio)
        pred_acc = calculate_accuracy(real_rul, predicted, train_ratio)
        print(f"Tool {tool_no} processing complete.")
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
        print(f"Error processing Tool {tool_no}: {e}")
        return {}


# -------------------- Additional Health Indicator Generation --------------------
def generate_trends(life_data: np.ndarray) -> List[np.ndarray]:
    """
    Generate six different trend indicators based on life data.
    (This is a placeholder function for adding additional comparative health indicators.)

    Parameters:
        life_data (np.ndarray): Life data array.

    Returns:
        List[np.ndarray]: List of trend indicator arrays.
    """
    # Placeholder: User can extend with actual health indicator computations.
    return []


# -------------------- Main Function --------------------
def main():
    """Main entry point of the program."""
    print("Program execution started...")

    # Genetic Algorithm (GA) Parameter Optimization
    max_generations = 1  # Set to 1 for testing; increase for full optimization
    population_size = 560
    mutation_rate = 0.3
    crossover_rate = 0.46

    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
        print("Starting parameter optimization via parallel Genetic Algorithm...")
        best_params, best_fit = parallel_genetic_algorithm(
            objective_function, BOUNDS, population_size, max_generations, mutation_rate, crossover_rate, executor
        )
        print(f"GA Optimization completed. Best Parameters: {best_params}, Best Fitness: {best_fit:.4f}")
        print("\nSkipping local optimization (polishing) steps.")

        # Process Data for Each Tool
        print(f"\nTools to be processed: {TOOL_NUMBERS}")
        tool_results = {}
        future_to_tool = {executor.submit(process_tool, tn, best_params): tn for tn in TOOL_NUMBERS}
        for fut in concurrent.futures.as_completed(future_to_tool):
            res = fut.result()
            if res:
                tool_results[res.get('tool_number')] = res
            else:
                print("Warning: A tool failed to process, skipping visualization.")
        print("All tool data processing complete.")

        # Visualization and Metric Calculation
        trend_labels = ['RS-HI', 'RMS-HI', 'ED-HI', 'GMM-HI', 'KLD-HI', 'MMD-HI']
        combined_metrics = []
        trends_all = {}
        diff_all = {}

        for tool_no in TOOL_NUMBERS:
            if tool_no not in tool_results:
                print(f"Warning: No valid result for Tool {tool_no}, skipping visualization and metrics.")
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
            trends = generate_trends(life_data)
            for trend, label in zip(trends, trend_labels):
                key = f'Tool_{tool_no}_{label}'
                trends_all[key] = trend
                diff_all[key] = np.abs(trend - perfect_trend)
            diff_all[f'Tool_{tool_no}_CF-HI'] = np.abs(predicted_rul - perfect_trend)

            visualize_results(
                life_data, shock_data, perfect_trend, predicted_rul, pulse_indices, pulse_line,
                failure_threshold, life_acc, pred_acc, tool_no,
                additional_trends=trends, additional_trend_labels=trend_labels,
                shock_upper=shock_upper, shock_lower=shock_lower
            )
            print(f"Visualization for Tool {tool_no} completed.")

            cfhi_metrics = calculate_trend_metrics(perfect_trend, predicted_rul)
            cfhi_metrics['Tool Number'] = tool_no
            cfhi_metrics['Model'] = 'CF-HI'
            combined_metrics.append(cfhi_metrics)

            for trend, label in zip(trends, trend_labels):
                trend_metrics = calculate_trend_metrics(perfect_trend, trend)
                trend_metrics['Tool Number'] = tool_no
                trend_metrics['Model'] = label
                combined_metrics.append(trend_metrics)

        # Save metrics, trends, and differences to an Excel file
        output_excel = os.path.join("../paper_results", "combined_metrics.xlsx")
        metrics_df = pd.DataFrame(combined_metrics)
        trends_df = pd.DataFrame(trends_all)
        differences_df = pd.DataFrame(diff_all)
        with pd.ExcelWriter(output_excel) as writer:
            # Convert wide metrics table to long format (melt)
            metrics_cols = ['MAE', 'MSE', 'Monotonicity', 'RMSE', 'Robustness', 'Trend']
            melted = pd.melt(
                metrics_df,
                id_vars=['Tool Number', 'Model'],
                value_vars=metrics_cols,
                var_name='Metric',
                value_name='Value'
            )
            # Pivot the melted table: rows = [Metric, Model], columns = Tool Number
            pivoted_metrics = melted.pivot_table(
                index=['Metric', 'Model'],
                columns='Tool Number',
                values='Value',
                aggfunc='mean'
            )
            # Reorder tool columns
            tool_order = [20104, 20106]
            pivoted_metrics = pivoted_metrics.reindex(columns=tool_order)
            # Construct MultiIndex for rows in desired order
            metric_order = ['MAE', 'MSE', 'Monotonicity', 'RMSE', 'Robustness', 'Trend']
            model_order = ['RS-HI', 'RMS-HI', 'ED-HI', 'GMM-HI', 'KLD-HI', 'MMD-HI', 'CF-HI']
            new_index = pd.MultiIndex.from_product([metric_order, model_order], names=['Metric', 'Model'])
            pivoted_metrics = pivoted_metrics.reindex(new_index)
            # Write metrics, trends, and differences to Excel sheets
            pivoted_metrics.to_excel(writer, sheet_name='Metrics')
            trends_df.to_excel(writer, sheet_name='Trends', index=False)
            differences_df.to_excel(writer, sheet_name='Differences', index=False)
        print(f"\nAll figures and metrics have been saved to {output_excel}")
        print("Program execution completed.")


if __name__ == "__main__":
    main()
