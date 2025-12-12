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
from train_helper import EarlyStopping, EarlyStoppingPercentage
from eval.evaluate_recon import evaluate_metrics, confusion_matrix_metrics
from preprocessing.preprocessing import * 
from sklearn.model_selection import train_test_split
from eval.evaluate_outliers import *
from models import BasicAutoencoder

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

def corrupt_data( X: Tensor, mask: Tensor, n_binary_cols: int) -> Tensor:
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
        noise = torch.randn_like(X[:, n_binary_cols:]) 
        X_corrupted[:, n_binary_cols:][num_mask] += noise[num_mask]
    
    return X_corrupted

def train_denoising_autoencoder(epochs=50, batch_size=256, learning_rate=1e-3,
                            hidden_dim_1=256, hidden_dim_2=64, latent_dim=32, patience = 2, 
                            min_delta = 0.005, num_weight=4):
    
    with open('synthetic_data/data/train_test_data_log.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    with open('synthetic_data/data/transformer_log.pkl', 'rb') as f:
        transformer = pickle.load(f)

    y_train = np.array(loaded_data['y_train'])
    y_test = np.array(loaded_data['y_test'])
    X_train = transformer.transform_input()
    X_test = transformer.transform_input_X(loaded_data['X_test'])

    input_dim = X_train.shape[1]
    top_n = y_train.sum()
    
    print(transformer.get_metadata())

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    n_binary_cols = len(transformer.get_OHEncoded_cols()) + len(transformer.bool_cols)
    model = BasicAutoencoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim, n_binary_cols)
    num_criterion = nn.MSELoss(reduction='none')
    bool_criterion = nn.BCELoss(reduction='none')
    bool_weight = 1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
   
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    save_train_loss = []
    save_num_losses = []
    save_binary_losses = []
    save_normal_loan_losses = []
    save_f1_train = []
    save_precision_train = []
    save_recall_train = []
    save_acc_train = []
    save_f1_test = []
    save_precision_test = []
    save_recall_test = []
    save_acc_test = []

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_losses = 0.0
        binary_losses = 0.0
        for batch in train_loader:
            x = batch[0]
            mask = get_random_corruption_mask(x)
            x_noisy = corrupt_data(x, mask, n_binary_cols)
            alpha = torch.distributions.Beta(0.5, 0.5).sample()
            num_loss, binary_loss, per_sample_loss = compute_denoising_per_sample_loss(model,x_noisy,n_binary_cols,num_weight,bool_weight, 
                                                                                                         alpha, mask, num_criterion, bool_criterion, training = True)
            loss = torch.mean(per_sample_loss) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_losses += num_loss.mean().item()
            binary_losses += binary_loss.mean().item()

        model.eval()
        with torch.no_grad():
            pred_labels_error_train, pred_labels_error_test = get_error_score_new_prediction_train_test(X_train, X_test, top_n, model, 
                                                            attributes_info=transformer.get_metadata(), cat_cols = transformer.cat_cols, bool_cols=transformer.bool_cols)
            normal_loan_losses = compute_per_sample_loss(model,X_train[y_train==0], n_binary_cols, num_weight, bool_weight, num_criterion, bool_criterion, training = False)
            normal_loan_losses = torch.mean(normal_loan_losses)

        print("\n")
        print(f"EPOCH [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.6f}")
        print(f"Num loss sum: {num_losses/len(train_loader):.4f}, Bool loss sum: {binary_losses/len(train_loader):.4f}")
        print(f"Normal loan losses: {normal_loan_losses:.4f}")
        print('TRAIN SET: ')
        f1_train, precision_train, recall_train, acc_train = evaluate_metrics(y_train, pred_labels_error_train)
        print('TEST SET: ')
        f1_test, precision_test, recall_test, acc_test = evaluate_metrics(y_test, pred_labels_error_test)

        save_train_loss.append(train_loss/len(train_loader))
        save_num_losses.append(num_losses/len(train_loader))
        save_binary_losses.append(binary_losses/len(train_loader))
        save_normal_loan_losses.append(normal_loan_losses)
        save_f1_train.append(f1_train)
        save_precision_train.append(precision_train)
        save_recall_train.append(recall_train)
        save_acc_train.append(acc_train)
        save_f1_test.append(f1_test)
        save_precision_test.append(precision_test)
        save_recall_test.append(recall_test)
        save_acc_test.append(acc_test)

        if early_stopping(normal_loan_losses, model, f1_train=f1_train, f1_test=f1_test):
            print(f"\nTraining stopped at epoch {epoch}")
            print(f"Loading best model with loss: {early_stopping.best_val_loss:.6f}")
            break

    best_model_state = early_stopping.best_model_state
    early_stopping_f1_train = early_stopping.best_f1_train
    early_stopping_f1_test = early_stopping.best_f1_test

    save_epoch_losses = {'train_loss': save_train_loss,
                    'num_loss': save_num_losses,
                    'bool_loss': save_binary_losses,
                    'normal_loan_loss' :save_normal_loan_losses,
                    'f1_train': f1_train,
                    'precision_train': precision_train,
                    'recall_train':recall_train,
                    'acc_train':acc_train,
                    'f1_test':f1_test,
                    'precision_test':precision_test,
                    'recall_test':recall_test,
                    'acc_test':acc_test,
                   }
    
    return model, best_model_state, save_epoch_losses, early_stopping_f1_train, early_stopping_f1_test


if __name__ == "__main__":
    hidden_dim_1 = 64
    hidden_dim_2 = 32
    latent_dim = 16
    learning_rate = 1e-5
    patience = 3
    min_delta = 0.005
    num_weight = 12

    model, best_model_state, save_epoch_losses, early_stopping_f1_train, early_stopping_f1_test = train_denoising_autoencoder(
        epochs=500,
        batch_size=128,
        learning_rate=learning_rate,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2, #64
        latent_dim=latent_dim, #32
        patience = patience,
        min_delta= min_delta,
        num_weight = num_weight
    )
    experiment_name = f'{hidden_dim_1}_{hidden_dim_2}_{latent_dim}_lr{learning_rate*10000}_nw{num_weight}_p{patience}_mindelta0'
    path = os.path.join("C:/Users/midon/Documents/anomaly-detection-autoencoder-based-basic/saved_models/hyperparams_tuning/autoencoder", experiment_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(model, path+ '/full_model.pth')
    # torch.save(model.state_dict(), path + "/weights.pth")
    torch.save(best_model_state, path + "/weights.pth")
    with open(path + '/epoch_losses.pkl', 'wb') as f:
        pickle.dump(save_epoch_losses, f)

    hyperparams = {
                'hidden_dim_1': hidden_dim_1,
                'hidden_dim_2': hidden_dim_2,
                'latent_dim': latent_dim,
                'learning_rate': learning_rate,
                'num_weight': num_weight,
                'patience': patience,
                'min_delta': min_delta}
    with open(path + '/hyperparams.pkl', 'wb') as f:
        pickle.dump(hyperparams, f)