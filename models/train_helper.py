from copy import deepcopy
import numpy as np 
import torch.nn as nn
import torch 
from typing import Union
from torch import Tensor
import random

def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class EarlyStopping:
    def __init__(self, patience=2, min_delta = 0.01):
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.min_delta = min_delta
        self.best_f1_train = float('inf')
        self.best_f1_test = float('inf')
        self.best_epoch = 0 

    def __call__(self, val_loss, model, f1_train, f1_test, epoch):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model_state = deepcopy(model.state_dict())
            self.best_f1_train = f1_train
            self.best_f1_test = f1_test
            self.best_epoch = epoch
            print(f"Loss improved to {val_loss:.6f} - Model saved")
            return False
        else:
            self.counter += 1
            print(f"No improvement - Counter: {self.counter}/{self.patience}")
          
            if self.counter >= self.patience:
                print(f"Early stopping triggered!")
                print(f"Best loss: {self.best_val_loss:.6f}")
                return True
        return False

class EarlyStoppingPercentage:
    def __init__(self, patience=2, min_delta=0.01, min_delta_type='absolute'):
    
        self.patience = patience
        self.min_delta = min_delta
        self.min_delta_type = min_delta_type
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        # Calculate dynamic threshold based on type
        if self.min_delta_type == 'percentage':
            threshold = self.best_val_loss * (1 - self.min_delta)  
        else:  # absolute
            threshold = self.best_val_loss - self.min_delta  
        
        if val_loss < threshold:
            improvement = self.best_val_loss - val_loss
            improvement_pct = (improvement / self.best_val_loss) * 100
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model_state = deepcopy(model.state_dict())
            print(f"Loss improved to {val_loss:.6f} ({improvement_pct:.2f}% improvement) - Model saved")
            return False
        else:
            self.counter += 1
            print(f"No significant improvement - Counter: {self.counter}/{self.patience}")
          
            if self.counter >= self.patience:
                print(f"Early stopping triggered!")
                print(f"Best loss: {self.best_val_loss:.6f}")
                return True
                
        return False
    
class EarlyStoppingNormalTest:
    def __init__(self, patience=2):
        self.patience = patience
        self.best_train_loss = float('inf')
        self.track_test_loss = float('inf')
        self.counter = 0
        self.counter_test_loss = 0 
        self.best_model_state = None
        
    def __call__(self, normal_train_loss, normal_test_loss, model):
        if (normal_train_loss < self.best_train_loss):
            self.best_train_loss = normal_train_loss
            self.track_test_loss = normal_test_loss
            self.counter = 0
            self.counter_test_loss = 0 
            self.best_model_state = deepcopy(model.state_dict())
            print(f"Loss improved to {normal_train_loss:.6f} - Model saved")
            return False
        else:
            self.counter += 1
            if normal_test_loss > self.track_test_loss: 
                self.counter_test_loss += 1
            print(f"No improvement - Counter: {self.counter}/{self.patience}")
          
            if (self.counter >= self.patience) & (self.counter_test_loss >=1):
                print(f"Early stopping triggered!")
                print(f"Best loss: {self.best_train_loss:.6f}")
                return True
        return False
    
# Early Stopping with when all errors decreases but normal train increases
class EarlyStoppingNew:
    def __init__(self, patience=2, min_epoch = None):
        self.patience = patience
        self.best_train_loss = float('inf')
        self.track_all_train_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.min_epoch = min_epoch
        
    def __call__(self, normal_train_loss, all_train_loss, model, epoch = None):
        if self.min_epoch is not None and epoch >= self.min_epoch:
            if (normal_train_loss < self.best_train_loss):
                self.best_train_loss = normal_train_loss
                self.track_all_train_loss = all_train_loss
                self.counter = 0
                self.best_model_state = deepcopy(model.state_dict())
                print(f"Loss improved to {normal_train_loss:.6f} - Model saved")
                return False
            else:
                self.counter += 1
                print(f"No improvement - Counter: {self.counter}/{self.patience}")
                # if all losses reduce but normal train does not reduce after n epochs, the model might learn to reduce losses of default loans
                if (self.counter >= self.patience) & (all_train_loss < self.track_all_train_loss):
                    print(f"Early stopping triggered!")
                    print(f"Best loss: {self.best_train_loss:.6f}")
                    return True
            return False
        
class EarlyStoppingWindowIncreases:
    def __init__(self, window_size=6, increase_threshold=3, min_delta = 0.01):
        """
        Early stopping based on counting loss increases in a sliding window.
        
        Args:
            window_size: Number of recent epochs to monitor
            increase_threshold: Stop if this many increases occur in the window
        """
        self.window_size = window_size
        self.increase_threshold = increase_threshold
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.min_delta = min_delta

    def __call__(self, val_loss, model):
        
        self.val_loss_history.append(val_loss)
        if len(self.val_loss_history) < self.window_size:
            return False
        
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_model_state = deepcopy(model.state_dict())
            print(f"Best validation loss: {val_loss:.6f} - Model saved")
        
        #based on instability, not fixed on consecutive epochs
        recent_losses = self.val_loss_history[-self.window_size:]
        increase_count = 0
        for i in range(1, len(recent_losses)):
            if recent_losses[i] > recent_losses[i-1]:
                increase_count += 1
        
        print(f"Increases in window: {increase_count}/{self.increase_threshold}")
        
        # Stop if increases >= threshold
        if increase_count >= self.increase_threshold:
            print(f"Early stopping triggered! {increase_count} increases in last {self.window_size} epochs")
            print(f"Best validation loss: {self.best_val_loss:.6f}")
            return True
        
        return False
    
def eval_scores_GAN(normal_data: Union[np.ndarray, Tensor], fake_data: Tensor):
    if isinstance(normal_data, np.ndarray):
        normal_data = torch.from_numpy(normal_data).float()

    real_mean = torch.mean(normal_data,dim=0)
    fake_mean = torch.mean(fake_data,dim=0)
    real_std = torch.std(normal_data,dim=0)
    fake_std = torch.std(fake_data,dim=0)

    distribution_error_mean = torch.mean((real_mean - fake_mean) ** 2).item()
    distribution_error_std = torch.mean((real_std - fake_std) ** 2).item()
    distribution_error_combined = distribution_error_mean + distribution_error_std
    return distribution_error_combined

def get_error_score_GAN(real_data:np.ndarray, fake_data:np.ndarray, attributes_info:dict, cat_cols:list, bool_cols:list) -> float:
    recon_errors = np.zeros_like(real_data)
    cat_set = set(cat_cols)
    bool_set = set(bool_cols)
    for i, (key, item) in enumerate(attributes_info.items()):
        start = item['start_index']
        end = start + item['n_values']
        if key in cat_set:
            vals = 1 - fake_data[:, start:end].max(axis=1)
        elif key in bool_set:
            vals = np.abs(real_data[:, start] - fake_data[:, start])
        else:
            diff = real_data[:, start] - fake_data[:, start]
            vals = 1 - np.exp(-(diff**2))
        recon_errors[:, i] = vals
    return recon_errors.mean()

def dynamic_steps_GAN(real_score_mean, fake_score_mean):
    if real_score_mean < 0.55 and fake_score_mean < 0.55:
        d_steps = 2
        g_steps = 1
        print("Need to strengthen discriminator (2D:1G)")
    
    # # If real scores too high 
    # elif real_score_mean > 0.75:
    #     d_steps = 1
    #     g_steps = 2
    #     print("Need to strengthen generator (1D:2G)")
    
    # If fake scores too low 
    elif fake_score_mean < 0.5 and real_score_mean > 0.6:
        g_steps = 3
        d_steps = 1
        print("Need to strengthen generator (1D:3G)")
    
    # If scores are in target range (0.6-0.7)
    elif 0.6 <= real_score_mean <= 0.7 and 0.6 <= fake_score_mean <= 0.7:
        d_steps = 1
        g_steps = 1
        print("OPTIMAL: Scores in target range 0.6-0.7 (1D:1G)")
    
    else:
        d_steps = 1
        g_steps = 1
        
    return d_steps, g_steps