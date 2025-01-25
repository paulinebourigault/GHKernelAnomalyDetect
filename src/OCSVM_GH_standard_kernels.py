"""
Kernel-Based Anomaly Detection with GH Kernels and Standard Kernels

This script demonstrates the use of Generalized Hyperbolic (GH) kernels and 
standard kernels (RBF, Polynomial, Linear, Sigmoid) for anomaly detection 
using One-Class SVM.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
from scipy.special import kv
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)

# Generate a synthetic dataset
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


# Generalized Hyperbolic (GH) Kernel function
def generalized_hyperbolic_kernel(X1, X2, params):
    lam, alpha, beta, delta, mu = (
        params["lambda"],
        params["alpha"],
        params["beta"],
        params["delta"],
        params["mu"],
    )
    gamma = np.sqrt(alpha**2 - beta**2)
    if alpha**2 <= beta**2:
        raise ValueError("Invalid GH parameters: alpha^2 must be > beta^2 for stability.")

    z = np.linalg.norm(X1[:, None] - X2, axis=2)
    bessel_term = kv(lam - 0.5, alpha * np.sqrt(delta**2 + z**2))
    bessel_term = np.nan_to_num(bessel_term, nan=1e-10, posinf=1e-10, neginf=1e-10)

    kernel_matrix = (
        (gamma / delta) ** lam
        / (np.sqrt(2 * np.pi) * kv(lam, delta * gamma))
        * np.exp(beta * z)
        * bessel_term
        * (np.sqrt(delta**2 + z**2) / alpha) ** (lam - 0.5)
    )
    return kernel_matrix

# Standard kernels
def rbf_kernel(X1, X2, gamma):
    return np.exp(-gamma * np.linalg.norm(X1[:, None] - X2, axis=2)**2)

def polynomial_kernel(X1, X2, degree, gamma, coef0):
    return (gamma * X1.dot(X2.T) + coef0) ** degree

def linear_kernel(X1, X2):
    return X1.dot(X2.T)

def sigmoid_kernel(X1, X2, gamma, coef0):
    return np.tanh(gamma * X1.dot(X2.T) + coef0)

# GH Kernel configurations
gh_kernel_configs = {
    "Full GH Kernel": {
        "lambda": [-1.0, 0.0, 1.0, 2.0],
        "alpha": [1.0, 1.5, 2.0, 2.5],
        "beta": [-0.5, 0.0, 0.5],
        "delta": [0.5, 1.0, 1.5],
        "mu": [-0.5, 0.0, 0.5],
    },
    "GH Kernel (Gaussian)": {
        "lambda": [0.5],
        "alpha": [2.0, 2.5, 3.0],  
        "beta": [0.0],
        "delta": [1.0, 1.5],      
        "mu": [0.0],
    },
    "GH Kernel (NIG)": {
        "lambda": [-0.5],
        "alpha": [2.0, 2.5, 3.0],
        "beta": [0.5, 0.8],
        "delta": [1.0, 1.5],
        "mu": [0.0],
    },
    "GH Kernel (Student's t)": {
        "lambda": [-1.0],
        "alpha": [2.0, 2.5, 3.0],
        "beta": [0.0],
        "delta": [1.0, 1.5],
        "mu": [0.0],
    },
    "GH Kernel (Hyperbolic)": {
        "lambda": [1.0],
        "alpha": [1.5, 2.0, 2.5],
        "beta": [0.3, 0.5],
        "delta": [1.0, 1.5],
        "mu": [0.0],
    },
}


def plot_decision_boundary(X, y, kernel_func, kernel_params, title, file_name):
    """
    Plot decision boundary for the trained One-Class SVM with separation of anomalies and normal data.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]

    if kernel_func == generalized_hyperbolic_kernel:
        K_test = kernel_func(grid, X, kernel_params)
    else:
        K_test = kernel_func(grid, X, **kernel_params)

    ocs = OneClassSVM(kernel="precomputed", nu=0.1)
    K_train = kernel_func(X, X, kernel_params) if kernel_func == generalized_hyperbolic_kernel else kernel_func(X, X, **kernel_params)
    ocs.fit(K_train)

    Z = ocs.decision_function(K_test).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.8)

    plt.contour(xx, yy, Z, levels=[0], colors="black", linewidths=2)

    plt.scatter(
        X[y == 1, 0], X[y == 1, 1], 
        label="Normal", color="blue", marker="o", alpha=0.7, edgecolor="k"
    )

    plt.scatter(
        X[y == -1, 0], X[y == -1, 1], 
        label="Anomaly", color="red", marker="^", alpha=0.9, edgecolor="k"
    )

    plt.legend()
    plt.title(title)
    plt.colorbar(label="Decision Function Value")
    plt.savefig(file_name)
    plt.close()


def plot_all_histograms(histograms, file_name):
    """
    Plot all histograms as subplots in a 3x3 grid with a visible decision boundary line.
    """
    num_kernels = len(histograms)
    rows = 3
    cols = int(np.ceil(num_kernels / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, (ax, (kernel_name, (decision_values, y_test))) in enumerate(zip(axes, histograms.items())):
        ax.hist(
            decision_values[y_test == 1],
            bins=30,
            alpha=0.7,
            label="Normal",
            color="blue",
        )
        ax.hist(
            decision_values[y_test == -1],
            bins=30,
            alpha=0.7,
            label="Anomaly",
            color="red",
        )

        # Add a visible decision boundary line at 0
        ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Decision Boundary (0)")

        ax.set_title(kernel_name, fontsize=14, fontweight="bold")

        if i // cols == rows - 1:  
            ax.set_xlabel("Decision Function Value", fontsize=12)
        else:
            ax.set_xticks([])

        if i % cols == 0:  
            ax.set_ylabel("Frequency", fontsize=12)
        else:
            ax.set_yticks([])

        ax.tick_params(axis="both", labelsize=10)
        
        if i == 0:  
            ax.legend(fontsize=10)

    
    for ax in axes[len(histograms):]:
        ax.remove()

    plt.savefig(file_name, dpi=300)
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_complex_data()

    kernel_methods = {
        "RBF": rbf_kernel,
        "Polynomial": polynomial_kernel,
        "Linear": linear_kernel,
        "Sigmoid": sigmoid_kernel,
        **{k: generalized_hyperbolic_kernel for k in gh_kernel_configs.keys()},
    }

    kernel_hyperparams = {
        **gh_kernel_configs,
        "RBF": {"gamma": [0.1]},
        "Polynomial": {"degree": [3], "gamma": [0.5], "coef0": [1]},
        "Linear": {},
        "Sigmoid": {"gamma": [0.5], "coef0": [1]},
    }

    results = []
    best_results = {}

    nu_values = [0.05, 0.1, 0.2, 0.3]  # Range of nu values to tune
    
    histograms = {}

    for kernel_name, kernel_func in kernel_methods.items():
        param_grid = ParameterGrid(kernel_hyperparams.get(kernel_name, [{}]))
        best_auc = 0
        best_params = None
        best_nu = None

        for nu in nu_values:
            for params in param_grid:
                try:
                    if kernel_name in gh_kernel_configs:
                        K_train = generalized_hyperbolic_kernel(X_train, X_train, params)
                        K_test = generalized_hyperbolic_kernel(X_test, X_train, params)
                    else:
                        K_train = kernel_func(X_train, X_train, **params)
                        K_test = kernel_func(X_test, X_train, **params)

                    ocs = OneClassSVM(kernel="precomputed", nu=nu)
                    ocs.fit(K_train)

                    y_scores = ocs.decision_function(K_test)
                    auc = roc_auc_score(y_test, y_scores)

                    if auc > best_auc:
                        best_auc = auc
                        best_params = params
                        best_nu = nu

                    results.append({
                        "Kernel": kernel_name,
                        "nu": nu,
                        "Parameters": params,
                        "AUC-ROC": auc
                    })

                except Exception as e:
                    logging.error(f"Error with kernel {kernel_name}, nu {nu}, and params {params}: {e}")
                    continue

        if best_params is not None:
            best_results[kernel_name] = {"Best AUC-ROC": best_auc, "Best Parameters": best_params, "Best nu": best_nu}
            plot_decision_boundary(
                np.vstack([X_train, X_test]),
                np.hstack([y_train, y_test]),
                kernel_func,
                best_params,
                f"{kernel_name} Kernel Decision Boundary (nu={best_nu})",
                f"{kernel_name}_decision_boundary_nu_{best_nu}.png"
            )
            
            decision_values = ocs.decision_function(K_test)
            histograms[kernel_name] = (decision_values, y_test)

    plot_all_histograms(histograms, "all_decision_function_histograms.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results_ocsvm_synthetic.csv", index=False)

    best_results_df = pd.DataFrame.from_dict(best_results, orient='index')
    best_results_df.to_csv("best_results_ocsvm_synthetic.csv")

    print("All Results:")
    print(results_df)
    print("Best Results:")
    print(best_results_df)