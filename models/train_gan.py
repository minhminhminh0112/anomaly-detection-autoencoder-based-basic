import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os 
from dotenv import load_dotenv
load_dotenv('.env.local')
PROJECT_PATH = os.getenv('PROJECT_PATH')
os.chdir(PROJECT_PATH)
import sys
sys.path.append(PROJECT_PATH)
from train_helper import get_error_score_GAN, eval_scores_GAN, EarlyStopping
from models import Generator, Discriminator
import pickle

def train_gan(epochs=200, batch_size=256, learning_rate_g=2e-4, learning_rate_d=2e-4, noise_dim=100,
              hidden_dim_g=128, hidden_dim_d=128, dropout = 0.2):
    """
    Train a GAN on transformed VehicleData.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizers
        noise_dim: Dimension of input noise vector
        hidden_dim_g: Smallest hidden layer dimension for generator
        hidden_dim_d: Smallest hidden layer dimension for discriminator
    """
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
    n_binary_cols = len(transformer.get_OHEncoded_cols()) + len(transformer.bool_cols)

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(n_binary_cols, noise_dim, input_dim, hidden_dim_g)
    discriminator = Discriminator(input_dim, hidden_dim_d, dropout =dropout)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g) # , betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d)#, betas=(0.5, 0.999)) # reduce learning rate if discriminator is too strong

    print(f"\nStarting GAN training for {epochs} epochs...")

    real_label = 0.9
    fake_label = 0.1

    early_stopping = EarlyStopping(patience = 10)
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        d_losses = []
        g_losses = []
        d_real_accs = []
        d_fake_accs = []

        for batch_idx, batch in enumerate(train_loader):
            real_data = batch[0]
            batch_size_current = real_data.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            real_labels = torch.full((batch_size_current, 1), real_label)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            noise = torch.randn(batch_size_current, noise_dim)
            fake_data = generator(noise)
            fake_labels = torch.full((batch_size_current, 1), fake_label)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            d_real_acc = (real_output > 0.5).float().mean().item() 
            d_fake_acc = (fake_output <= 0.5).float().mean().item()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            noise = torch.randn(batch_size_current, noise_dim)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)

            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_g.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_real_accs.append(d_real_acc)
            d_fake_accs.append(d_fake_acc)

        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        avg_d_real_acc = np.mean(d_real_accs)
        avg_d_fake_acc = np.mean(d_fake_accs)

        # if (epoch + 1) % 10 == 0:
        discriminator.eval()
        generator.eval()
        
        with torch.no_grad():
            normal_train_data = X_train[y_train == 0] #compare error with the fake data
            noise = torch.randn(len(normal_train_data), noise_dim)
            fake_data = generator(noise)
            real_scores = discriminator(torch.FloatTensor(normal_train_data))
            fake_scores = discriminator(fake_data)
            
            mean_error = get_error_score_GAN(real_data=normal_train_data, fake_data = fake_data.detach().numpy(), attributes_info = transformer.get_metadata(), cat_cols=transformer.cat_cols, bool_cols =transformer.bool_cols )
            # MONITOR: if the fake data gets same scores as normal real then 0 
            score_diff = torch.abs(real_scores.mean() - fake_scores.mean())
            distribution_error_combined = eval_scores_GAN(normal_train_data, fake_data)

        print('\n')
        print(f"Epoch [{epoch+1}/{epochs}]")
        # MONITOR: D Loss and G Loss are stable
        print(f"  D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}") 
        print(f"  D Real Acc: {avg_d_real_acc:.2%}, D Fake Acc: {avg_d_fake_acc:.2%}")
        # this tells us what is the average sigmoid value for real and fake 
        print(f'  Real scores: {real_scores.mean():.4f}')
        print(f'  Fake scores: {fake_scores.mean():.4f}')
        print(f'  Difference between real and fake scores:  {score_diff:.4f}')
        # MONITOR the data distribution of fake and real data, should be 0.1 or lower
        print(f'  Distribution error:  {distribution_error_combined}') 
        print(f'  Mean differences between real and fake (all cells):   {mean_error}')
    # add early stopping: high real score and small gap 
        # if early_stopping(mean_error, generator):
        #     print(f"\nTraining stopped at epoch {epoch}")
        #     print(f"Loading best model with loss: {early_stopping.best_val_loss:.6f}")
        #     break
        # best_model_state = early_stopping.best_model_state
    return generator, discriminator, transformer

if __name__ == "__main__":

    epochs=40
    batch_size=128
    learning_rate_g=1e-3 #2e-4
    learning_rate_d=1e-3
    noise_dim=100
    hidden_dim_g=256
    hidden_dim_d=32
    dropout = 0.2

    generator, discriminator, transformer = train_gan(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate_g=learning_rate_g, #2e-4
        learning_rate_d=learning_rate_d,
        noise_dim=noise_dim,
        hidden_dim_g=hidden_dim_g,
        hidden_dim_d=hidden_dim_d,
        dropout = dropout
    )
    experiment_name = 'first_gan_log'
    path = os.path.join("C:/Users/midon/Documents/anomaly-detection-autoencoder-based-basic/saved_models", experiment_name)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(generator, path+ '/generator.pth')
    torch.save(generator.state_dict(), path + '/generator_weights.pth')
    torch.save(discriminator, path+ '/discriminator.pth')
    torch.save(discriminator.state_dict(), path + '/discriminator_weights.pth')

    hyperparams = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate_g': learning_rate_g, #2e-4
        'learning_rate_d': learning_rate_d,
        'noise_dim': noise_dim,
        'hidden_dim_g': hidden_dim_g,
        'hidden_dim_d': hidden_dim_d,
        'dropout': dropout
        }
    with open(path + '/hyperparams.pkl', 'wb') as f:
        pickle.dump(hyperparams, f)