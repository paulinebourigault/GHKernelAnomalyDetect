"""
Variational Autoencoder (VAE) for Anomaly Detection

This script implements a Variational Autoencoder (VAE) to detect anomalies in a synthetic non-Gaussian dataset.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import itertools


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


# Step 2: Model Definition
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model with customizable hidden and latent dimensions.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # latent_dim * 2 for mean and logvar
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        params = self.encoder(x)
        mean, logvar = torch.chunk(params, 2, dim=1)  # Split the output into mean and logvar
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar


# Step 3: Loss Function
def vae_loss_function(reconstructed, original, mean, logvar):
    """
    Compute the loss for the Variational Autoencoder.
    Combines reconstruction loss and KL divergence.
    """
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence / original.size(0) 


# Step 4: Training and Evaluation
def train_vae(X_train, X_test, y_test, hidden_dims, latent_dims, learning_rates, num_epochs_list):
    """
    Train and evaluate the Variational Autoencoder using grid search over hyperparameters.
    Returns:
        best_auc: Best AUC-ROC score
        best_params: Parameters corresponding to the best AUC-ROC
        results_df: DataFrame containing results for all configurations
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    best_auc = 0
    best_params = {}
    results = []

    for hidden_dim, latent_dim, lr, num_epochs in itertools.product(hidden_dims, latent_dims, learning_rates, num_epochs_list):
        # Initialize the model
        model = VAE(input_dim=X_train.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                batch_data = batch[0]
                reconstructed, mean, logvar = model(batch_data)
                loss = vae_loss_function(reconstructed, batch_data, mean, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            reconstructed_test, _, _ = model(X_test_tensor)
            reconstructed_test = reconstructed_test.numpy()
            reconstruction_error = ((X_test_tensor.numpy() - reconstructed_test) ** 2).sum(axis=1)

            auc = roc_auc_score(y_test, reconstruction_error)

            if auc > best_auc:
                best_auc = auc
                best_params = {"hidden_dim": hidden_dim, "latent_dim": latent_dim, "learning_rate": lr, "num_epochs": num_epochs}

            results.append({
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "learning_rate": lr,
                "num_epochs": num_epochs,
                "AUC-ROC": auc
            })

    results_df = pd.DataFrame(results)
    return best_auc, best_params, results_df


# Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()

    hidden_dims = [16, 32, 64]
    latent_dims = [2, 4, 6, 8]
    learning_rates = [0.001, 0.01]
    num_epochs_list = [20, 30, 50]

    best_auc, best_params, results_df = train_vae(
        X_train, X_test, y_test, hidden_dims, latent_dims, learning_rates, num_epochs_list
    )

    print("Best AUC-ROC:", best_auc)
    print("Best Parameters:", best_params)
    print(results_df)

    results_df.to_csv("vae_anomaly_results.csv", index=False)