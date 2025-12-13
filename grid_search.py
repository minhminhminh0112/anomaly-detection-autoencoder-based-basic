from sklearn.model_selection import ParameterGrid
# from models.train_autoencoder import train_basic_autoencoder
from models.train_denoising import train_denoising_autoencoder
import pandas as pd 
import pickle
import os 
import torch

param_grid = {
    'hidden_dim_1': [128, 64],
    'hidden_dim_2': [64, 32],
    'latent_dim': [16, 8],
    'learning_rate': [1e-4, 1e-5],
    'num_weight': [4, 8, 12]
}
# Grid search
results = []
best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Testing: {params}")
    model, best_model_state, save_epoch_losses, early_stopping_f1_train, early_stopping_f1_test, early_stopping_epoch = train_denoising_autoencoder(
        epochs=600,
        batch_size=128,
        learning_rate=params['learning_rate'],
        hidden_dim_1=params['hidden_dim_1'],
        hidden_dim_2=params['hidden_dim_2'], #64
        latent_dim=params['latent_dim'], #32
        patience = 3,
        min_delta= 0.005,
        num_weight = params['num_weight']
    )
    
    result = params.copy()
    result['epochs'] = early_stopping_epoch
    result['f1_train'] = early_stopping_f1_train
    result['f1_test'] = early_stopping_f1_test
    results.append(result)
    
    experiment_name = f'{params['hidden_dim_1']}_{params['hidden_dim_2']}_{params['latent_dim']}_lr{float(params['learning_rate'])*10000}_nw{params['num_weight']}_p3_mindelta005'
    path = os.path.join("C:/Users/midon/Documents/anomaly-detection-autoencoder-based-basic/saved_models/hyperparams_tuning/denoising", experiment_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(model, path+ '/full_model.pth')
    # torch.save(model.state_dict(), path + "/weights.pth")
    torch.save(best_model_state, path + "/weights.pth")
    with open(path + '/epoch_losses.pkl', 'wb') as f:
        pickle.dump(save_epoch_losses, f)

    with open(path + '/hyperparams.pkl', 'wb') as f:
        pickle.dump(params, f)


df_results = pd.DataFrame(results)
df_results.to_csv(path + 'grid_search_results.csv', index=False)
df_results.to_csv('C:/Users/midon/Documents/anomaly-detection-autoencoder-based-basic/saved_models/hyperparams_tuning/denoising/'+'grid_search_results.csv', index=False)