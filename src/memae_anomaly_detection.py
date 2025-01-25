"""
Memory-Augmented Autoencoder (MemAE) for Anomaly Detection on synthetic non-Gaussian dataset
Dong Gong et al., “Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection,” in ICCV, 2019.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
import pandas as pd
from sklearn.model_selection import train_test_split


# Dataset Preparation
def generate_data(n_samples=1000, anomaly_ratio=0.05, noise=0.1):
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


class MemoryModule(nn.Module):
    def __init__(self, memory_size, memory_dim):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))  # Memory bank
        nn.init.xavier_uniform_(self.memory)

    def forward(self, x):
        attention_weights = torch.softmax(torch.matmul(x, self.memory.t()), dim=-1)
        memory_output = torch.matmul(attention_weights, self.memory)
        return memory_output, attention_weights


class MemAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size, memory_dim):
        super(MemAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim)
        )
        self.memory = MemoryModule(memory_size, memory_dim)
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_memory, attention_weights = self.memory(z)
        x_hat = self.decoder(z_memory)
        return x_hat, z_memory, attention_weights


def train_memae(model, train_loader, optimizer, criterion, n_epochs=50):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            x_hat, _, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss / len(train_loader):.4f}")


def evaluate_memae(model, test_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            x_hat, _, _ = model(x)
            reconstruction_error = torch.mean((x_hat - x) ** 2, dim=1)
            scores.extend(reconstruction_error.cpu().numpy())
    return np.array(scores)


param_grid = {
    "hidden_dim": [16, 32, 64],
    "memory_size": [20, 50, 100],
    "memory_dim": [8, 16, 32],
    "learning_rate": [0.0001, 0.001, 0.01],
    "n_epochs": [50]
}

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=16, shuffle=False)

    best_auc = 0
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        print(f"Evaluating params: {params}")
        model = MemAE(
            input_dim=X_train.shape[1],
            hidden_dim=params["hidden_dim"],
            memory_size=params["memory_size"],
            memory_dim=params["memory_dim"]
        )
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.MSELoss()

        train_memae(model, train_loader, optimizer, criterion, n_epochs=params["n_epochs"])

        anomaly_scores = evaluate_memae(model, test_loader)
        auc = roc_auc_score(y_test, anomaly_scores)
        print(f"AUC-ROC: {auc:.4f}")

        results.append({
            "params": params,
            "AUC-ROC": auc
        })

        if auc > best_auc:
            best_auc = auc
            best_params = params

    print(f"Best AUC-ROC: {best_auc:.4f} with params: {best_params}")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("memae_anomaly_results.csv", index=False)