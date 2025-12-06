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
from copy import deepcopy
from eval.evaluate_recon import evaluate_metrics, confusion_matrix_metrics
from preprocessing.preprocessing import * 
from sklearn.model_selection import train_test_split
from eval.evaluate_outliers import *
from train_helper import EarlyStopping
from models import BasicAutoencoder
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def train_basic_autoencoder(epochs=50, batch_size=256, learning_rate=1e-3,
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
    
    X_transformed = final_data.get_X_train(array_format = True)
    y = final_data.y
    transformer = final_data.transform(final_data.X_raw)
    print(final_data.X_raw.shape)
    print(transformer.get_metadata())
    # data = np.load('data_removed_iso_outliers.npz')
    # X_transformed = data['X_transformed']
    # y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )
    y_train = np.array(y_train)
    input_dim = X_train.shape[1]
    top_n = y_train.sum()
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    n_binary_cols = len(transformer.get_OHEncoded_cols()) + len(transformer.bool_cols)
    model = BasicAutoencoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim, n_binary_cols)
    num_criterion = nn.MSELoss(reduction='none')
    bool_criterion = nn.BCELoss(reduction='none')
    bool_weight = 1 #n_binary_cols / input_dim
    num_weight = 10

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
   
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    early_stopping = EarlyStopping(patience=3)
    best_fraud_rate = 0.0
    best_epoch = 0 
    best_error_score_fraud_rate = 0.0
    best_error_score_epoch = 0 
    # epoch_pred_labels = {}
    # epoch_pred_labels['real_labels'] = y_train
    # epoch_pred_labels['epochs'] = {}
    save_train_loss = []
    save_num_losses = []
    save_binary_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_losses = 0.0
        binary_losses = 0.0
        for batch in train_loader:

            x = batch[0]
            num_loss, binary_loss, per_sample_loss = compute_per_sample_loss(model,x,n_binary_cols,num_weight,bool_weight, num_criterion, bool_criterion, training = True)
            loss = torch.mean(per_sample_loss)
                            
            num_losses += num_loss.mean().item()
            binary_losses += binary_loss.mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            pred_labels_train, pred_labels_test = get_loss_prediction_train_test(X_train, X_test, top_n, model, 
                                                                                 n_binary_cols, num_weight, bool_weight, num_criterion, 
                                                                                 bool_criterion)
            
            pred_labels_error_train, pred_labels_error_test = get_error_score_prediction_train_test(X_train, X_test, top_n, model, 
                                                                                 n_binary_cols)
            normal_loan_losses = compute_per_sample_loss(model,X_train[y_train==0], n_binary_cols, num_weight, bool_weight, num_criterion, bool_criterion, training = False)
            normal_loan_losses = torch.mean(normal_loan_losses)
            # epoch_pred_labels['epochs'][epoch] = {
            #     'pred_labels': pred_labels.copy(),
            #     'num_loss': num_loss.cpu().numpy().copy(),
            #     'binary_loss': binary_loss.cpu().numpy().copy(),
            #     'per_sample_loss': per_sample_loss
            # }
        save_train_loss.append(train_loss)
        save_num_losses.append(num_losses/len(train_loader))
        save_binary_losses.append(binary_losses/len(train_loader))
        print("\n")
        print(f"EPOCH [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")
        print(f"Num loss sum: {num_losses/len(train_loader):.4f}, Bool loss sum: {binary_losses/len(train_loader):.4f}")
        print('TRAIN SET: ')
        loss, detected_fraud_rate, recall, acc = evaluate_metrics(y_train, pred_labels_train)
        loss_error, detected_fraud_rate_error, recall_error, acc_error = evaluate_metrics(y_train, pred_labels_error_train)
        print('TEST SET: ')
        evaluate_metrics(y_test, pred_labels_test)
        loss_error_test, detected_fraud_rate_error_test, recall_error_test, acc_error_test = evaluate_metrics(y_test, pred_labels_error_test)

        if detected_fraud_rate_error > best_error_score_fraud_rate:
            best_error_score_fraud_rate = detected_fraud_rate_error
            best_error_score_epoch = epoch
        print(f'best precision based on error at epoch {best_error_score_epoch +1} TRAIN: {best_error_score_fraud_rate:.2%}')

        if detected_fraud_rate_error_test > best_fraud_rate:
            best_fraud_rate = detected_fraud_rate_error_test
            best_epoch = epoch
        print(f'best precision based on error at epoch {best_epoch +1} TEST: {best_fraud_rate:.2%}')

        if early_stopping(normal_loan_losses, model):
            print(f"\nTraining stopped at epoch {epoch}")
            print(f"Loading best model with loss: {early_stopping.best_loss:.6f}")
            best_model_state = early_stopping.best_model_state
            break

    save_epoch_losses = {'train_loss': save_train_loss,
                   'num loss': save_num_losses,
                   'bool loss': save_binary_losses,
                   'len train loader': len(train_loader)}
    
    return best_model_state, transformer, y, save_epoch_losses


if __name__ == "__main__":
    # Train the model
    best_model_state, transformer, y, save_epoch_losses = train_basic_autoencoder(
        epochs=200,
        batch_size=128,
        learning_rate=1e-4,
        hidden_dim_1=64,
        hidden_dim_2=16, #64
        latent_dim=8 #32
    )
    experiment_name = '20 final features'
    path = os.path.join("C:/Users/midon/Documents/anomaly-detection-autoencoder-based-basic/saved_models", experiment_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # torch.save(model.state_dict(), path + "/weights.pth")
    torch.save(best_model_state, path + "/weights.pth")
    with open(path + 'epoch_losses.pkl', 'wb') as f:
        pickle.dump(save_epoch_losses, f)

    with open(path + '/transformer.pkl', 'wb') as f:
        pickle.dump(transformer, f)

    np.save(path + "/y.npy", y)