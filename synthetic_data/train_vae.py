import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd 
import pickle 
import mlflow
import mlflow.pytorch
import os
import random
import sys

# Add the src directory to Python path so we can import from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae import VariationalAutoencoder, loss_function
import preprocessing.preprocessing as pp

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

experiment_name = "first_vae_experiment"

# Configurations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
scaler_type = 'standard'  # 'standard' or 'minmax'
kl_weights =1
cat_weight = 1
cont_weight = 5 # increase to 6,8,10 
latent_dims = 32 #input_dims 52
layers_encoder = np.array([128, 64])
layers_decoder = np.array([64, 128])
pre_cont_layer = 32 # try 16 increase cat loss 360 after 100 epochs 
pre_cat_layer = 32 # try 16 
latent_dist = 'normal'
reduction = 'sum'
batch_size = 128
epochs = 400
learning_rate = 1e-4  # Reduced learning rate for more stable training

# Import and transform data
vehicle= pp.VehicleData(train_mode=True)
transformer = vehicle.transform(vehicle.X_raw)
X_transformed = vehicle.get_X_train(array_format=True, scaler_type=scaler_type)
y = vehicle.y
cat_dims = X_transformed.shape[1]-(len(vehicle.num_cols) + len(vehicle.date_cols)) #42

train_loader = DataLoader(dataset = X_transformed,
                                    batch_size = batch_size,
                                    worker_init_fn=seed_worker,
                                    generator=g)
model = VariationalAutoencoder(latent_dims= latent_dims, latent_dist = latent_dist,
                                                input_dims= X_transformed.shape[1],
                                                cat_dims=cat_dims,
                                                layers_encoder= layers_encoder, layers_decoder=layers_decoder,
                                                pre_cont_layer=pre_cont_layer, pre_cat_layer= pre_cat_layer)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
categorical_losses = []
numerical_losses = []
kl_losses = []

def train():
    model.train()    
        
    for epoch in range(epochs):
        # gradient_norms = 0 
        sum_cat_losses = 0 
        sum_cont_losses = 0 
        sum_kl_losses = 0 
        for x_batch in train_loader:
            optimizer.zero_grad()
            cat_output, cont_output, alpha, beta = model(x_batch)
            cat_loss,cont_loss,kl_loss= loss_function(x = x_batch, cat_x_hat = cat_output, 
                                            cont_x_hat=cont_output, alpha = alpha, 
                                            beta=beta, prior = latent_dist, reduction=reduction)
            loss = cat_weight*cat_loss + cont_weight*cont_loss + kl_weights* kl_loss
            loss.backward()

            optimizer.step()
            sum_cat_losses += cat_loss.item()
            sum_cont_losses += cont_loss.item()
            sum_kl_losses += kl_loss.item()
            
        epoch_loss = sum_cat_losses + sum_cont_losses + sum_kl_losses
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} (Cat: {sum_cat_losses/len(train_loader):.4f}, Cont: {sum_cont_losses/len(train_loader):.4f}, KL: {sum_kl_losses/len(train_loader):.4f})")

        mlflow.log_metric("loss", round(avg_loss, 4), step=epoch)
        mlflow.log_metric("cat_loss", round(sum_cat_losses / len(train_loader),4), step=epoch)
        mlflow.log_metric("cont_loss", round(sum_cont_losses / len(train_loader),4), step=epoch)
        mlflow.log_metric("kl_loss", round(sum_kl_losses / len(train_loader),4), step=epoch)

if __name__ == "__main__":
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    path = os.path.join("saved_models", experiment_name)

    os.makedirs(path, exist_ok=True)
    mlflow.set_tracking_uri("mlruns") 
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "input_dims": X_transformed.shape[1],
            "latent_dims": latent_dims,
            "cat_dims": cat_dims,
            "layers_encoder": layers_encoder, 
            "layers_decoder": layers_decoder, 
            "pre_cont_layer": pre_cont_layer,
            "pre_cat_layer": pre_cat_layer,
            "latent_dist": latent_dist,
            "kl_weights": kl_weights,
            "cat_weight": cat_weight,
            "cont_weight": cont_weight,
            "scaler_type": scaler_type,
            "reduction": reduction,
        })
        train()
        
        mlflow.pytorch.log_model(model, "model")
        
        # Create the files first, then log them as artifacts
        torch.save(model.state_dict(), path + "/weights.pth")
        np.save(path + "/X_samples.npy", X_transformed[:51])
        with open(path + "/model_architecture.txt", "w") as f:
            f.write(str(model))
        with open(path + '/transformer.pkl', 'wb') as f:
            pickle.dump(transformer, f)
        # Now log the artifacts (files must exist first)
        mlflow.log_artifact(path + "/transformer.pkl")
        mlflow.log_artifact(path + "/weights.pth")
        mlflow.log_artifact(path + "/X_samples.npy")
        mlflow.log_artifact(path + "/model_architecture.txt")

        print(f"Run completed: {run.info.run_id}")
