import os 
from dotenv import load_dotenv
load_dotenv('.env.local')
PROJECT_PATH = os.getenv('PROJECT_PATH')
os.chdir(PROJECT_PATH)
import sys
sys.path.append(PROJECT_PATH)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from data_loading_utils import load_and_preprocess_vehicle_data
from eval.evaluate_recon import evaluate_metrics, confusion_matrix_metrics

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BasicAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(BasicAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def train_basic_autoencoder(epochs=50, batch_size=256, learning_rate=1e-3,
                            hidden_dim=64, latent_dim=32):
    """
    Train a basic autoencoder on transformed VehicleData using MSELoss.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
    """

    X_train, X_val, y_train, y_val, transformer = load_and_preprocess_vehicle_data() #path = 'rf_rfe_selected_features.json'
    #     test_size=0.2,
    #     random_state=42,
    #     scaler_type='standard'
    # )

    input_dim = X_train.shape[1]

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BasicAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
    # Separate losses for numerical and boolean features
    num_criterion = nn.MSELoss()
    bool_criterion = nn.BCEWithLogitsLoss()
    n_bool_cols = len(transformer.bool_cols)
    print('bool cols: ',transformer.bool_cols)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_fraud_rate = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            # Forward pass
            reconstructed = model(x)
            num_loss = num_criterion(reconstructed[:,n_bool_cols:], x[:,n_bool_cols:])
            bool_loss = bool_criterion(reconstructed[:,:n_bool_cols], x[:,:n_bool_cols])
            loss = num_loss + bool_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        # Fraud detection analysis every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")
            model.eval()
            with torch.no_grad():
                # Calculate per-sample losses on training data
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                reconstructed_train = model(X_train_tensor)
                per_sample_loss = torch.mean((reconstructed_train - X_train_tensor) ** 2, dim=1) #MSE works best, instead of separating losses for bool and num
                per_sample_loss = per_sample_loss.cpu().numpy()
                n_fraud_labels = y_train.sum()

                top_loss_idx = np.argsort(per_sample_loss)[-n_fraud_labels:]
                top_losses = per_sample_loss[top_loss_idx]
                top_loss_labels = y_train[top_loss_idx]

                fraud_count = top_loss_labels.sum()
                detected_fraud_rate = fraud_count / n_fraud_labels

                print(f"      Top {n_fraud_labels} highest loss samples: {fraud_count} frauds ({detected_fraud_rate:.1%})")
                print(f"      Mean loss in top {n_fraud_labels}: {top_losses.mean():.6f}")
                print(f"      Loss range: [{top_losses.min():.6f}, {top_losses.max():.6f}]")
                pred_labels = np.zeros_like(y_train)
                pred_labels[top_loss_idx] = 1
                evaluate_metrics(y_train, pred_labels)
                confusion_matrix_metrics(y_train, pred_labels)
        # Save best model
        if detected_fraud_rate < best_fraud_rate:
            best_fraud_rate = detected_fraud_rate
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'detected_fraud_rate': detected_fraud_rate,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'latent_dim': latent_dim
            }, 'basic_autoencoder_best.pth')

    print(f"\nTraining complete! Best fraud rate: {best_fraud_rate:.6f}")
    print("Model saved as 'basic_autoencoder_best.pth'")

    return model, transformer


if __name__ == "__main__":
    # Train the model
    model, transformer = train_basic_autoencoder(
        epochs=200,
        batch_size=256,
        learning_rate=1e-3,
        hidden_dim=128,
        latent_dim=16
    )
