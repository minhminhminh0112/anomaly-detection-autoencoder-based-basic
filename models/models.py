import torch.nn as nn 
import torch 

class BasicAutoencoder(nn.Module):
    """
    A simple autoencoder with sigmoid as last activation layer for boolean cols.
    """
    def __init__(self, input_dim, hidden_dim_1 = 256, hidden_dim_2=64, latent_dim=32, n_binary_cols:int=3):
        super(BasicAutoencoder, self).__init__()
        self.n_binary_cols = n_binary_cols

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

        self.sigmoid = nn.Sigmoid() # for binary cols
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        if self.n_binary_cols > 0:
            reconstructed_binary = self.sigmoid(reconstructed[:, :self.n_binary_cols])
            reconstructed_continuous = reconstructed[:, self.n_binary_cols:]
            reconstructed_concat = torch.cat([reconstructed_binary, reconstructed_continuous], dim=1)
        return reconstructed_concat

class Generator(nn.Module):
    """
    Generator network that takes Gaussian random noise and generates synthetic vehicle data.
    """
    def __init__(self, noise_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, z):
        """
        Args:
            z: Random noise tensor of shape (batch_size, noise_dim)
        Returns:
            Generated data of shape (batch_size, output_dim)
        """
        return self.network(z)


class Discriminator(nn.Module):
    """
    Discriminator network that distinguishes between real vehicle data and generated fake data.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Input data of shape (batch_size, input_dim)
        Returns:
            Probability that input is real (batch_size, 1)
        """
        return self.network(x)