"""
This script implements the Deep Support Vector Data Description (Deep SVDD) framework for anomaly detection
on a synthetic dataset (Ruff et al., “Deep one-class classification,” in ICML, 2018)

Author: Pauline Bourigault
Date: 27/11/2024
"""

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Dataset Generation
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

# Step 2: Deep SVDD Model Definition
class DeepSVDD(nn.Module):
    """
    Deep Support Vector Data Description (Deep SVDD) Model
    """
    def __init__(self, input_dim, hidden_dim):
        super(DeepSVDD, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)

# Step 3: Center Initialization
def initialize_center(deep_svdd, train_loader):
    """
    Initialize the hypersphere center as the mean of the network's outputs on the training data.
    """
    for layer in reversed(deep_svdd.network):
        if isinstance(layer, nn.Linear):
            c = torch.zeros(layer.out_features)
            break
    else:
        raise ValueError("No linear layer found in the Deep SVDD network.")

    n_samples = 0
    deep_svdd.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, _ = batch
            z = deep_svdd(x)
            n_samples += z.size(0)
            c += z.sum(dim=0)
    c /= n_samples
    return c

# Step 4: Training Function
def train_deep_svdd(model, center, train_loader, n_epochs=50, lr=1e-3):
    """
    Train the Deep SVDD model using the specified number of epochs and learning rate.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            x, _ = batch
            optimizer.zero_grad()
            z = model(x)
            # Deep SVDD objective: minimize distance to center
            loss = torch.mean(torch.sum((z - center) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader)

# Step 5: Evaluation Function
def evaluate_deep_svdd(model, center, test_loader):
    """
    Evaluate the Deep SVDD model using AUC-ROC.
    """
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            z = model(x)
            dist = torch.sum((z - center) ** 2, dim=1)  # Squared distance from the center
            scores.extend(dist.numpy())
            labels.extend(y.numpy())

    scores = np.array(scores)
    labels = np.array(labels)

    auc = roc_auc_score(labels, -scores)  # Negative distance to center
    return auc

# Step 6: Hyperparameter Grid Search
def perform_grid_search(X_train, X_test, y_train, y_test):
    """
    Perform grid search over hidden dimensions, learning rates, and epochs.
    """
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    hidden_dim_grid = [16, 32, 64]
    lr_grid = [1e-3, 1e-4]
    epoch_grid = [20, 30, 50]

    best_auc = 0
    best_params = None

    for hidden_dim, lr, n_epochs in itertools.product(hidden_dim_grid, lr_grid, epoch_grid):
        # Initialize the model and center
        model = DeepSVDD(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
        center = initialize_center(model, train_loader)

        train_deep_svdd(model, center, train_loader, n_epochs=n_epochs, lr=lr)
        auc = evaluate_deep_svdd(model, center, test_loader)

        if auc > best_auc:
            best_auc = auc
            best_params = {"hidden_dim": hidden_dim, "lr": lr, "n_epochs": n_epochs}

    print("Best AUC-ROC:", best_auc)
    print("Best Parameters:", best_params)

# Step 7: Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()
    perform_grid_search(X_train, X_test, y_train, y_test)
