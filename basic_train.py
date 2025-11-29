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
from eval.evaluate_recon import evaluate_metrics, confusion_matrix_metrics
from preprocessing.preprocessing import * 
from sklearn.model_selection import train_test_split
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BasicAutoencoder(nn.Module):
    """
    A simple autoencoder with MSELoss for reconstructing transformed VehicleData.
    """
    def __init__(self, input_dim,hidden_dim_1 = 256, hidden_dim_2=64, latent_dim=32):
        super(BasicAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def train_basic_autoencoder(epochs=50, batch_size=256, learning_rate=1e-3,
                            hidden_dim_1=256, hidden_dim_2=64, latent_dim=32):
    """
    Train a basic autoencoder on transformed VehicleData using MSELoss.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
    """
    path = 'synthetic_data/synthetic_data.csv'
    synth = SyntheticData(path)
    # synth.bool_cols
    final_data = feature_engineer(synth)
    X_transformed = final_data.get_X_train(array_format = True)
    y = final_data.y
    transformer = final_data.transform(final_data.X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )
    y_train = np.array(y_train)
    input_dim = X_train.shape[1]

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BasicAutoencoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim).to(device)
    num_criterion = nn.MSELoss(reduction='none')
    bool_criterion = nn.BCEWithLogitsLoss(reduction='none')
    n_bool_cols = len(transformer.bool_cols)
    bool_weight = 1 #n_bool_cols / input_dim
    num_weight = 8

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
   
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_fraud_rate = 0.0
    best_epoch = 0 

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_losses = 0.0
        bool_losses = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            reconstructed = model(x)
            num_loss = num_criterion(reconstructed[:,n_bool_cols:], x[:,n_bool_cols:])
            bool_loss = bool_criterion(reconstructed[:,:n_bool_cols], x[:,:n_bool_cols])

            num_losses += num_loss.mean().item()
            bool_losses += bool_loss.mean().item()

            num_loss_sum = torch.sum(num_loss, dim=1)  
            bool_loss_sum = torch.sum(bool_loss, dim=1) 
            loss = torch.mean(num_weight * num_loss_sum + bool_weight * bool_loss_sum)
                            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            # Calculate per-sample losses on training data
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            reconstructed_train = model(X_train_tensor)
            
            num_loss = num_criterion(reconstructed_train[:,n_bool_cols:], X_train_tensor[:,n_bool_cols:])
            bool_loss = bool_criterion(reconstructed_train[:,:n_bool_cols], X_train_tensor[:,:n_bool_cols])
            
            num_loss_sum = torch.sum(num_loss, dim=1) #  
            bool_loss_sum = torch.sum(bool_loss, dim=1) 
            
            per_sample_loss = num_weight * num_loss_sum + bool_weight * bool_loss_sum
            per_sample_loss = per_sample_loss.cpu().numpy()
            n_fraud_labels = y_train.sum()

            top_loss_idx = np.argsort(per_sample_loss)[-n_fraud_labels:]
            top_loss_labels = y_train[top_loss_idx]

            fraud_count = top_loss_labels.sum()
            detected_fraud_rate = fraud_count / n_fraud_labels

            pred_labels = np.zeros_like(y_train)
            pred_labels[top_loss_idx] = 1
            
        print("\n")
        print(f"EPOCH [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")
        print(f"Num loss sum: {num_losses/len(train_loader):.4f}, Bool loss sum: {bool_losses/len(train_loader):.4f}")
        print(f"      Top {n_fraud_labels} highest loss samples: {fraud_count} frauds ({detected_fraud_rate:.1%})")
        
        evaluate_metrics(y_train, pred_labels)
        # confusion_matrix_metrics(y_train, pred_labels)
        if detected_fraud_rate > best_fraud_rate:
            best_fraud_rate = detected_fraud_rate.copy()
            best_epoch = epoch
        print(f'best detected fraud rate at epoch {best_epoch}: {best_fraud_rate:.2%}')
        
    return model, transformer


if __name__ == "__main__":
    # Train the model
    model, transformer = train_basic_autoencoder(
        epochs=200,
        batch_size=256,
        learning_rate=1e-4,
        hidden_dim_1=256,
        hidden_dim_2=64,
        latent_dim=16
    )
