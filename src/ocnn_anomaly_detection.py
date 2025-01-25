"""
OC-NN for Anomaly Detection

This script implements an OC-NN (One-Class Neural Network) for anomaly detection on a synthetic non-Gaussian dataset.
It performs a grid search over hyperparameters to find the best configuration based on AUC-ROC.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate a complex synthetic dataset
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


class OCNN(nn.Module):
    """
    One-Class Neural Network (OC-NN) model for anomaly detection.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(OCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def ocnn_loss(output, nu):
    """
    Compute OC-NN centered hypersphere loss.
    Args:
        output: Model output
        nu: Hyperparameter for anomaly detection
    Returns:
        Loss value
    """
    term1 = 0.5 * torch.sum(output**2)
    term2 = 1 / nu * torch.mean(torch.relu(1 - output))
    term3 = -torch.mean(output)
    return term1 + term2 + term3

def train_ocnn(model, train_loader, optimizer, nu, n_epochs):
    """
    Train OC-NN model.
    Args:
        model: OC-NN model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        nu: Hyperparameter for anomaly detection
        n_epochs: Number of training epochs
    """
    model.train()
    for epoch in range(n_epochs):
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = ocnn_loss(output, nu)
            loss.backward()
            optimizer.step()

def evaluate_ocnn(model, test_loader):
    """
    Evaluate OC-NN model.
    Args:
        model: Trained OC-NN model
        test_loader: DataLoader for test data
    Returns:
        Array of anomaly scores
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            output = model(x)
            scores.extend(output.squeeze().numpy())
    return np.array(scores)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    param_grid = {
        "hidden_dim": [8, 16, 32],
        "lr": [0.001, 0.01],
        "nu": [0.1, 0.2],
        "n_epochs": [20, 30, 50]
    }

    best_auc = 0
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        model = OCNN(input_dim=X_train.shape[1], hidden_dim=params["hidden_dim"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        
        train_ocnn(model, train_loader, optimizer, nu=params["nu"], n_epochs=params["n_epochs"])
        
        scores = evaluate_ocnn(model, test_loader)
        auc = roc_auc_score(y_test, -scores)  # Negative scores for anomaly detection
        
        results.append({"Parameters": params, "AUC-ROC": auc})
        
        if auc > best_auc:
            best_auc = auc
            best_params = params

    results_df = pd.DataFrame(results)
    print(f"Best Parameters: {best_params}")
    print(f"Best AUC-ROC: {best_auc:.4f}")
    print("\nFull Results:")
    print(results_df.sort_values(by="AUC-ROC", ascending=False))
