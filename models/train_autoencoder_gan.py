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
from models import Generator, BasicAutoencoder
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_autoencoder_from_fake_data(experiment_name:str,epochs=50, batch_size=256, learning_rate=1e-3,
                            hidden_dim_1=256, hidden_dim_2=64, latent_dim=32, n_fake_data = 100000):

    base_path = 'saved_models'
    path = os.path.join(base_path, experiment_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment path not found: {path}")
    with open('synthetic_data/data/transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    with open('synthetic_data/data/train_test_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    with open(os.path.join(path, 'hyperparams.pkl'),'rb') as r:
        hyperparams = pickle.load(r)

    y_train = np.array(loaded_data['y_train'])
    y_test = np.array(loaded_data['y_test'])
    X_train = transformer.transform_input()
    X_test = transformer.transform_input_X(loaded_data['X_test'])

    n_binary_cols = len(transformer.get_OHEncoded_cols()) + len(transformer.bool_cols)
    input_dim = len(transformer.input_cols)
    top_n = len(y_train[y_train==1])
    generator_model_state = torch.load(os.path.join(path, "generator_weights.pth"))
    generator = Generator(n_binary_cols = n_binary_cols, noise_dim = hyperparams['noise_dim'], output_dim=input_dim, hidden_dim = hyperparams['hidden_dim_g'])
    generator.load_state_dict(generator_model_state)
    generator.eval()
    noise = torch.randn(n_fake_data, hyperparams['noise_dim'])
    with torch.no_grad():
        fake_data = generator(noise)

    train_dataset = TensorDataset(torch.FloatTensor(fake_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
    early_stopping = EarlyStopping(patience=4, min_delta=0.0)
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
           
        save_train_loss.append(train_loss/len(train_loader))
        save_num_losses.append(num_losses/len(train_loader))
        save_binary_losses.append(binary_losses/len(train_loader))
        print("\n")
        print(f"EPOCH [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.6f}")
        print(f"Num loss sum: {num_losses/len(train_loader):.4f}, Bool loss sum: {binary_losses/len(train_loader):.4f}")
        print('TRAIN SET: ')
        loss, detected_fraud_rate, recall, acc = evaluate_metrics(y_train, pred_labels_train)
        loss_error, detected_fraud_rate_error, recall_error, acc_error = evaluate_metrics(y_train, pred_labels_error_train)
        print('TEST SET: ')
        evaluate_metrics(y_test, pred_labels_test)
        loss_error_test, detected_fraud_rate_error_test, recall_error_test, acc_error_test = evaluate_metrics(y_test, pred_labels_error_test)

        if early_stopping(train_loss/len(train_loader), model):
            print(f"\nTraining stopped at epoch {epoch}")
            print(f"Loading best model with loss: {early_stopping.best_val_loss:.6f}")
            break
    best_model_state = early_stopping.best_model_state
    save_epoch_losses = {'train_loss': save_train_loss,
                   'num loss': save_num_losses,
                   'bool loss': save_binary_losses,
                   'len train loader': len(train_loader)}
    
    return model, best_model_state, save_epoch_losses


if __name__ == "__main__":
    generator_experiment_name = 'first_gan'
    hidden_dim_1 = 128
    hidden_dim_2 = 64
    latent_dim = 16
    learning_rate = 1e-4
    
    model, best_model_state, save_epoch_losses = train_autoencoder_from_fake_data(
        experiment_name=generator_experiment_name,
        epochs=500,
        batch_size=128,
        learning_rate=learning_rate,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2, #64
        latent_dim=latent_dim #32
    )
    save_experiment_name = 'first_autoencoder_gan_first_gan'
    path = os.path.join("C:/Users/midon/Documents/anomaly-detection-autoencoder-based-basic/saved_models", save_experiment_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(model, path+ '/full_model.pth')
    # torch.save(model.state_dict(), path + "/weights.pth")
    torch.save(best_model_state, path + "/weights.pth")
    with open(path + '/epoch_losses.pkl', 'wb') as f:
        pickle.dump(save_epoch_losses, f)

    hyperparams = {
                'generator_experiment_name': generator_experiment_name,
               'hidden_dim_1': hidden_dim_1,
               'hidden_dim_2': hidden_dim_2,
               'latent_dim': latent_dim}
    with open(path + '/hyperparams.pkl', 'wb') as f:
        pickle.dump(hyperparams, f)