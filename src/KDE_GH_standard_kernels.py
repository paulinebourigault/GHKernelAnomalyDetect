"""
Kernel-Based Anomaly Detection using GH and Standard Kernels

This script implements kernel density estimation (KDE) for anomaly detection using:
- Generalized Hyperbolic (GH) Kernels (Gaussian, NIG, Student's t, Hyperbolic)
- Standard KDE Kernels (Gaussian, Tophat, Exponential, Epanechnikov)

Author: Pauline Bourigault (Modified)
Date: 27/11/2024
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
from scipy.special import kv
import matplotlib.pyplot as plt
import time

# Step 1: Dataset Generation
def generate_complex_data(n_samples=1000, anomaly_ratio=0.05, noise=0.1):
    """
    Generate a synthetic dataset with structured normal data on manifolds and sparse anomalies.
    """
    np.random.seed(42)

    # Number of anomalies and normal samples
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normals = n_samples - n_anomalies

    # Generate normal data: points on three sinusoidal manifolds
    t1 = np.linspace(0, 2 * np.pi, n_normals // 3)
    t2 = np.linspace(0, 2 * np.pi, n_normals // 3)
    t3 = np.linspace(0, 2 * np.pi, n_normals - 2 * (n_normals // 3))

    X_normal_1 = np.column_stack((t1, np.sin(t1) + np.random.normal(scale=noise, size=t1.shape)))
    X_normal_2 = np.column_stack((t2 + 2 * np.pi, np.cos(t2) + np.random.normal(scale=noise, size=t2.shape)))
    X_normal_3 = np.column_stack((-t3 - 2 * np.pi, np.sin(t3) + 2 + np.random.normal(scale=noise, size=t3.shape)))

    X_normal = np.vstack([X_normal_1, X_normal_2, X_normal_3])
    y_normal = np.ones(X_normal.shape[0])  # Normal class: 1

    # Generate anomalies: sparse points scattered randomly
    X_anomalies = np.random.uniform(low=-6, high=6, size=(n_anomalies, 2))
    y_anomalies = -np.ones(X_anomalies.shape[0])  # Anomaly class: -1

    # Combine normal data and anomalies
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([y_normal, y_anomalies])

    # Shuffle and split
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Step 2: Kernel Definitions
def generalized_hyperbolic_kernel(x, y, params):
    """
    Compute the Generalized Hyperbolic (GH) kernel value between two points.
    """
    lam, alpha, beta, delta, mu = (
        params["lambda"],
        params["alpha"],
        params["beta"],
        params["delta"],
        params["mu"],
    )
    gamma = np.sqrt(alpha**2 - beta**2)
    z = np.linalg.norm(x - y)

    bessel = kv(lam - 0.5, alpha * np.sqrt(delta**2 + z**2))
    kernel_value = (
        (gamma / delta) ** lam
        / (np.sqrt(2 * np.pi) * kv(lam, delta * gamma))
        * np.exp(beta * z)
        * bessel
        * (np.sqrt(delta**2 + z**2) / alpha) ** (lam - 0.5)
    )
    return kernel_value

def gaussian_kde(x, X, bandwidth):
    """Gaussian KDE kernel."""
    return np.sum(np.exp(-np.linalg.norm(x - X, axis=1)**2 / (2 * bandwidth**2))) / (np.sqrt(2 * np.pi) * bandwidth * len(X))

def tophat_kde(x, X, bandwidth):
    """Tophat KDE kernel."""
    return np.sum(np.linalg.norm(x - X, axis=1) < bandwidth) / (len(X) * bandwidth)

def exponential_kde(x, X, bandwidth):
    """Exponential KDE kernel."""
    return np.sum(np.exp(-np.linalg.norm(x - X, axis=1) / bandwidth)) / (len(X) * bandwidth)

def epanechnikov_kde(x, X, bandwidth):
    """Epanechnikov KDE kernel."""
    distances = np.linalg.norm(x - X, axis=1)
    return np.sum((1 - (distances / bandwidth)**2) * (distances < bandwidth)) / (len(X) * bandwidth)

def plot_decision_boundary(X_train, X_test, y_test, kernel_func, kernel_params, title, file_name):
    """
    Plot decision boundary for KDE kernel-based anomaly detection.
    """
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Compute decision function for grid points
    scores = np.array([-np.log(max(kernel_func(x, X_train, kernel_params), 1e-10)) for x in grid])
    Z = scores.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.8)
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="blue", label="Normal", edgecolor="k", marker="o", alpha=0.7)
    plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], c="red", label="Anomaly", edgecolor="k", marker="^", alpha=0.9)
    plt.colorbar(label="Decision Function Value")
    plt.title(title)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

# Step 3: Anomaly Detection Functions
def anomaly_score_kde(X, x, params, kernel_func, bandwidth):
    """
    Compute the anomaly score for a point using KDE with a given kernel.
    """
    density = np.mean([kernel_func(x, xi, params) if params else kernel_func(x, xi, bandwidth) for xi in X])
    return -np.log(max(density, 1e-10))  # Avoid log(0)

def evaluate_kernels(X_train, X_test, y_train, y_test, gh_kernel_configs, standard_kernels, bandwidths):
    """
    Evaluate GH and standard kernels, measure AUC-ROC, and training time.
    Returns:
        results: List of dictionaries containing kernel evaluation results
    """
    results = []

    # Evaluate GH Kernels
    for kernel_name, base_params in gh_kernel_configs.items():
        best_auc = 0
        best_params = None
        start_time = time.time()

        param_grid = ParameterGrid({
            "lambda": [base_params["lambda"] - 0.5, base_params["lambda"], base_params["lambda"] + 0.5],
            "alpha": [base_params["alpha"] * 0.8, base_params["alpha"], base_params["alpha"] * 1.2],
            "beta": [base_params["beta"] - 0.2, base_params["beta"], base_params["beta"] + 0.2],
            "delta": [base_params["delta"] * 0.8, base_params["delta"], base_params["delta"] * 1.2],
            "mu": [base_params["mu"]],
        })

        for params in param_grid:
            y_scores = [anomaly_score_kde(X_train, x, params, generalized_hyperbolic_kernel, bandwidth=None) for x in X_test]
            auc = roc_auc_score(y_test, -np.array(y_scores))

            if auc > best_auc:
                best_auc = auc
                best_params = params

        training_time = time.time() - start_time
        results.append({
            "Kernel": kernel_name,
            "Best AUC-ROC": best_auc,
            "Best Parameters": best_params,
            "Training Time (s)": training_time,
        })

    # Evaluate Standard KDE Kernels
    for kernel_name, kernel_func in standard_kernels.items():
        best_auc = 0
        best_params = None
        start_time = time.time()

        for bandwidth in bandwidths:
            y_scores = [kernel_func(x, X_train, bandwidth) for x in X_test]
            anomaly_scores = -np.log(np.maximum(y_scores, 1e-10))  # Avoid log(0)

            auc = roc_auc_score(y_test, anomaly_scores)

            if auc > best_auc:
                best_auc = auc
                best_params = {"bandwidth": bandwidth}

        training_time = time.time() - start_time
        results.append({
            "Kernel": kernel_name,
            "Best AUC-ROC": best_auc,
            "Best Parameters": best_params,
            "Training Time (s)": training_time,
        })
        
        # Plot decision boundary
        plot_decision_boundary(
            X_train,
            X_test,
            y_test,
            kernel_func,
            bandwidth,
            f"{kernel_name} (BW={bandwidth}) Decision Boundary",
            f"{kernel_name}_bw_{bandwidth}_decision_boundary.png"
        )

    return results

# Step 4: Main Execution
if __name__ == "__main__":
    # Generate dataset
    X_train, X_test, y_train, y_test = generate_complex_data()

    # Define GH kernel configurations
    gh_kernel_configs = {
        "Full GH Kernel": {"lambda": 1.0, "alpha": 2.0, "beta": 0.5, "delta": 1.0, "mu": 0.0},
        "GH Kernel (Gaussian)": {"lambda": 0.5, "alpha": 2.0, "beta": 0.0, "delta": 1.0, "mu": 0.0},
        "GH Kernel (NIG)": {"lambda": -0.5, "alpha": 2.0, "beta": 0.5, "delta": 1.0, "mu": 0.0},
        "GH Kernel (Student's t)": {"lambda": -1.0, "alpha": 2.0, "beta": 0.0, "delta": 1.0, "mu": 0.0},
        "GH Kernel (Hyperbolic)": {"lambda": 1.0, "alpha": 1.5, "beta": 0.3, "delta": 1.0, "mu": 0.0},
    }

    # Define standard KDE kernels
    standard_kernels = {
        "Gaussian KDE": gaussian_kde,
        "Tophat KDE": tophat_kde,
        "Exponential KDE": exponential_kde,
        "Epanechnikov KDE": epanechnikov_kde,
    }

    # Evaluate kernels
    bandwidths = [0.01, 0.1, 0.5]  # Bandwidth grid for standard kernels
    results = evaluate_kernels(X_train, X_test, y_train, y_test, gh_kernel_configs, standard_kernels, bandwidths)

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("evaluation_results_kde_synthetic.csv", index=False)