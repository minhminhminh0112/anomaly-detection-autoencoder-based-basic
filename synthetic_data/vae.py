import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from torch.special import digamma, gammaln
torch.manual_seed(42)
np.random.seed(42)

class Encoder(nn.Module):
    def __init__(self,latent_dims, input_dims, layers_size: np.array, latent_dist:str = 'normal'):
        super(Encoder, self).__init__()
        self.latent_dist = latent_dist
        self.n_layers = len(layers_size)
        self.layers = nn.ModuleList() 
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dims, layers_size[i]))
            else:
                self.layers.append(nn.Linear(layers_size[i-1], layers_size[i]))
        
        self.alpha_linear = nn.Linear(layers_size[-1], latent_dims)
        self.beta_linear = nn.Linear(layers_size[-1], latent_dims)

    def forward(self, x):
        if self.n_layers >=1:
            for layer in self.layers:
                x = F.relu(layer(x))#, negative_slope=0.4
        if self.latent_dist == 'gamma':
            alpha = torch.nn.functional.softplus(self.alpha_linear(x))
            beta = torch.nn.functional.softplus(self.beta_linear(x))
        else:
            #last layer should not be activated 
            alpha = self.alpha_linear(x)
            beta = self.beta_linear(x)
        return alpha, beta

class Decoder(nn.Module):
    def __init__(self, latent_dims,input_dims, cat_dims , layers_size: np.array, pre_cont_layer: int, pre_cat_layer: int):
        super(Decoder, self).__init__()
        self.n_layers = len(layers_size)
        self.layers = nn.ModuleList() 
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(nn.Linear(latent_dims, layers_size[i]))
            else:
                self.layers.append(nn.Linear(layers_size[i-1], layers_size[i]))
        
        self.cat_layer1 = nn.Linear(layers_size[-1],pre_cat_layer)
        # self.cat_layer2 = nn.Linear(128,64)
        self.cat_output_layer = nn.Linear(pre_cat_layer, cat_dims)

        self.cont_layer1 = nn.Linear(layers_size[-1],pre_cont_layer)
        # self.cont_layer2 = nn.Linear(32,16)
        self.cont_output_layer = nn.Linear(pre_cont_layer, input_dims-cat_dims)

    def forward(self, x):
        if self.n_layers >=1:
            for i in range(len(self.layers)):
                x = torch.relu(self.layers[i](x)) #torch.nn.LeakyReLU(negative_slope=0.4)(self.layers[i](x))
        cat_hidden_1 = torch.relu(self.cat_layer1(x))
        # cat_hidden_2 = torch.relu(self.cat_layer2(cat_hidden_1))
        cat_output = torch.sigmoid(self.cat_output_layer(cat_hidden_1)) #must use sigmoid for binary crossentropy

        cont_hidden_1 = torch.relu(self.cont_layer1(x))
        # cont_hidden_2= torch.relu(self.cont_layer2(cont_hidden_1))
        # cont_hidden_3= torch.relu(self.cont_layer3(cont_hidden_2))
        cont_output = self.cont_output_layer(cont_hidden_1)
        return cat_output,cont_output
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims:int, latent_dist:str, layers_encoder: np.array,layers_decoder: np.array, input_dims:int, cat_dims:int, pre_cont_layer: int, pre_cat_layer: int):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims = latent_dims,input_dims = input_dims, layers_size=layers_encoder, latent_dist=latent_dist)
        self.decoder = Decoder(latent_dims = latent_dims,input_dims = input_dims, cat_dims=cat_dims, layers_size=layers_decoder,pre_cont_layer=pre_cont_layer, pre_cat_layer=pre_cat_layer)
        self.latent_dist = latent_dist

    def reparameterization_normal(self, mean, var):
        epsilon = torch.randn_like(var)        # mean = 0 and variance=1 sampling epsilon        
        z = mean + torch.exp(0.5 * var)*epsilon   # kl vanishing: var equals 0, mean equals 0
        #z = mean + var * epsilon
        return z
    
    def reparameterization_gamma(self, alpha, beta):
        # Sample from a standard Gamma distribution with shape parameter alpha and rate=1
        # Make sure rate is positive
        gamma = torch.distributions.Gamma(alpha, beta)  # Define Gamma distribution
        z = gamma.rsample()  # Sample using reparameterization trick
        return z
    
    def forward(self, x):
        enc_output1, enc_output2 = self.encoder(x)
        if self.latent_dist == 'gamma':
            z = self.reparameterization_gamma(enc_output1, enc_output2)
        else:
            z = self.reparameterization_normal(enc_output1, enc_output2)
        cat_output,cont_output = self.decoder(z)
        if self.training:
            return cat_output,cont_output,enc_output1, enc_output2
        return cat_output,cont_output, z

def loss_function(x, cat_x_hat, cont_x_hat, alpha, beta, prior:str, reduction:str ='mean'):
    # x_hat[:,:end_cat_vars] = nn.Sigmoid()(x_hat[:,:end_cat_vars])
    # cat_preds = (x_hat[:,:end_cat_vars]>0.5).float()
    # Add loss for bool nn.BCEWithLogitsLoss(reduction="none")
    end_cat_vars = cat_x_hat.shape[1]
    cat_loss = F.binary_cross_entropy(cat_x_hat, x[:,:end_cat_vars], reduction=reduction) #before cross entropy, softmax
    cont_loss = nn.MSELoss(reduction= reduction)(cont_x_hat, x[:,end_cat_vars:])
    
    if prior == 'normal':
        # kl_loss = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        kl_loss = -0.5 * torch.sum(1 + beta - alpha.pow(2) - beta.exp()) 
    elif prior == 'gamma':
        alpha_p = torch.ones_like(alpha)  # Prior shape (1)
        beta_p = torch.ones_like(beta)   # Prior rate (1)

        # KL Divergence for Gamma(alpha_q, beta_q) || Gamma(1, 1)
        kl_loss = (alpha - alpha_p) * digamma(alpha) \
                - (gammaln(alpha) - gammaln(alpha_p)) \
                + alpha_p * (torch.log(beta) - torch.log(beta_p)) \
                + alpha * (beta_p / beta - 1)
        kl_loss = torch.sum(kl_loss)    
    else:
        raise ValueError('Invalid prior distribution')
    
    #kl_loss = torch.mean(kl_loss)
    if reduction == 'none':
        cat_loss = torch.mean(torch.sum(cat_loss, dim = 1))
        cont_loss = torch.mean(torch.sum(cont_loss, dim = 1)) #sum across columns --> mean
        kl_loss = torch.sum(kl_loss, dim=1)

    return cat_loss, cont_loss, kl_loss