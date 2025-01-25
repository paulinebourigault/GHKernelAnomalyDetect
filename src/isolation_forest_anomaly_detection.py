"""
Isolation Forest (Fei Tony Liu et al., “Isolation forest,” in IEEE ICDM, 2008, pp. 413–422)

This script performs anomaly detection using Isolation Forest on a synthetic non-Gaussian dataset.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
import pandas as pd

def generate_dataset(n_samples=1000, anomaly_ratio=0.05, noise=0.1):
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

def grid_search_isolation_forest(X_train, X_test, y_train, y_test, random_seeds):
    """
    Perform grid search on Isolation Forest with multiple seeds for robust evaluation.
    Args:
        X_train, X_test, y_train, y_test: Train-test splits
        random_seeds: List of random seeds for multi-seed evaluation

    Returns:
        results_df: DataFrame containing grid search results
        best_params: Best parameters based on Mean AUC-ROC
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_samples": [0.5, 1.0],
        "contamination": [0.1, 0.2],  # Percentage of anomalies
        "max_features": [1.0, 0.8],
    }

    all_results = []
    for params in ParameterGrid(param_grid):
        seed_results = []
        for seed in random_seeds:
            model = IsolationForest(**params, random_state=seed)
            model.fit(X_train)

            y_scores = -model.decision_function(X_test)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 1, -1)  # Align labels to 1 (normal), -1 (anomalous)

            auc = roc_auc_score(y_test, y_scores)
            seed_results.append(auc)

        mean_auc = np.mean(seed_results)
        std_auc = np.std(seed_results)

        all_results.append({
            "Parameters": params,
            "Mean AUC-ROC": mean_auc,
            "Std AUC-ROC": std_auc,
        })

    results_df = pd.DataFrame(all_results)

    # Sort by Mean AUC-ROC
    results_df.sort_values(by="Mean AUC-ROC", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Identify best parameters
    best_row = results_df.iloc[0]
    best_params = best_row["Parameters"]

    return results_df, best_params

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset()

    random_seeds = [42, 123, 456, 789, 101112]
    results_df, best_params = grid_search_isolation_forest(X_train, X_test, y_train, y_test, random_seeds)

    print(f"Best Parameters: {best_params}")
    best_row = results_df.iloc[0]
    print(f"Best Mean AUC-ROC: {best_row['Mean AUC-ROC']:.4f} ± {best_row['Std AUC-ROC']:.4f}")
    print("\nFull Results:")
    print(results_df)