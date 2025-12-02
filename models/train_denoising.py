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
from eval.evaluate_outliers import *
from train_autoencoder import BasicAutoencoder
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_random_corruption_mask(X):
    mask = np.zeros_like(X)
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    n_rows_to_change = int(0.03 * n_rows)
    rows_to_change = np.random.choice(n_rows, size=n_rows_to_change, replace=False)

    n_ones = int(0.5 * n_cols)

    for row_idx in rows_to_change:
        cols_to_mask = np.random.choice(n_cols, size=n_ones, replace=False)
        mask[row_idx, cols_to_mask] = 1
    return torch.from_numpy(mask)

def corrupt_data( X: Tensor, mask: Tensor, n_binary_cols: int, noise_std: Tensor) -> Tensor:
    """
    Corrupt data for denoising autoencoder.
    
    Args:
        X: Original data array (n_samples, n_features)
        mask: Binary mask indicating where to corrupt (n_samples, n_features)
        n_binary_cols: Number of binary columns 
        noise_std: Standard deviation of Gaussian noise for numerical features
    
    Returns:
        Corrupted data array
    """
    X_corrupted = X.clone()
    
    # For boolean attributes, flip value
    if n_binary_cols > 0:
        bool_mask = mask[:, :n_binary_cols].bool()
        X_corrupted[:, :n_binary_cols][bool_mask] = 1 - X_corrupted[:, :n_binary_cols][bool_mask]
    
    # For numerical attributes, add Gaussian noise
    if n_binary_cols < X.shape[1]:
        num_mask = mask[:, n_binary_cols:].bool()
        noise = torch.randn_like(X[:, n_binary_cols:]) * noise_std
        X_corrupted[:, n_binary_cols:][num_mask] += noise[num_mask]
    
    return X_corrupted

def train_denoising_autoencoder(epochs=50, batch_size=256, learning_rate=1e-3,
                            hidden_dim_1=256, hidden_dim_2=64, latent_dim=32, n_binary_cols = 3):
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
    # # synth.bool_cols
    final_data = feature_engineer(synth)
    # X_transformed = final_data.get_X_train(array_format = True)
    # y = final_data.y
    transformer = final_data.transform(final_data.X_raw)
    data = np.load('data_removed_iso_outliers.npz')
    X_transformed = data['X_transformed']
    
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )
    y_train = np.array(y_train)
    input_dim = X_train.shape[1]

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    n_binary_cols = len(transformer.get_OHEncoded_cols()) + len(transformer.bool_cols)
    model = BasicAutoencoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim, n_binary_cols)
    num_criterion = nn.MSELoss(reduction='none')
    bool_criterion = nn.BCELoss(reduction='none')
    bool_weight = 1 #n_binary_cols / input_dim
    num_weight = 8
    noise_std = torch.from_numpy(np.std(X_transformed[:,n_binary_cols:], axis = 0)).float()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
   
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_fraud_rate = 0.0
    best_epoch = 0 
    # epoch_pred_labels = {}
    # epoch_pred_labels['real_labels'] = y_train
    # epoch_pred_labels['epochs'] = {}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_losses = 0.0
        binary_losses = 0.0
        for batch in train_loader:
            x = batch[0]
            mask = get_random_corruption_mask(x)
            x_noisy = corrupt_data(x, mask, n_binary_cols,noise_std)
            alpha = torch.distributions.Beta(0.5, 0.5).sample()
            num_loss, binary_loss, per_sample_loss = compute_denoising_per_sample_loss(model,x_noisy,n_binary_cols,num_weight,bool_weight, 
                                                                                                         alpha, mask, num_criterion, bool_criterion, training = True)
            loss = alpha * torch.mean(per_sample_loss) + (1-alpha) * torch.mean(per_sample_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_losses += num_loss.mean().item()
            binary_losses += binary_loss.mean().item()

        # model.eval()
        # with torch.no_grad():
        #     # Calculate per-sample losses on training data
        #     per_sample_loss_train = compute_per_sample_loss(model,X_train,n_binary_cols,num_weight,bool_weight, num_criterion, bool_criterion, training=False)
        #     per_sample_loss_test = compute_per_sample_loss(model,X_test,n_binary_cols,num_weight,bool_weight, num_criterion, bool_criterion, training=False)
        #     top_n = y_train.sum()
        #     pred_labels_train = get_top_n_prediction(per_sample_loss_train, top_n)
    
        #     threshold_value = np.sort(per_sample_loss_train)[-top_n].item()
        #     print(threshold_value)
        #     pred_labels_test = get_threshold_prediction(per_sample_loss_test,threshold_value)
            # epoch_pred_labels['epochs'][epoch] = {
            #     'pred_labels': pred_labels.copy(),
            #     'num_loss': num_loss.cpu().numpy().copy(),
            #     'binary_loss': binary_loss.cpu().numpy().copy(),
            #     'per_sample_loss': per_sample_loss
            # }
            
        print("\n")
        print(f"EPOCH [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")
        print(f"Num loss sum: {num_losses/len(train_loader):.4f}, Bool loss sum: {binary_losses/len(train_loader):.4f}")
        print('TRAIN SET: ')
        # f1, detected_fraud_rate, recall, acc = evaluate_metrics(y_train, pred_labels_train)
        # print('TEST SET: ')
        # evaluate_metrics(y_test, pred_labels_test)
        # if detected_fraud_rate > best_fraud_rate:
        #     best_fraud_rate = detected_fraud_rate
        #     best_epoch = epoch
        # print(f'best detected fraud rate at epoch {best_epoch}: {best_fraud_rate:.2%}')
    # with open('epoch_pred_labels.pkl', 'wb') as f:
    #         pickle.dump(epoch_pred_labels, f)
    return model, transformer


if __name__ == "__main__":
    # Train the model
    model, transformer = train_denoising_autoencoder(
        epochs=200,
        batch_size=128,
        learning_rate=1e-5,
        hidden_dim_1=256,
        hidden_dim_2=64,
        latent_dim=16
    )
