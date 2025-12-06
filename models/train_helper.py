from copy import deepcopy

class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience
        self.best_train_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, normal_train_loss, model):
        if (normal_train_loss < self.best_train_loss):
            self.best_train_loss = normal_train_loss
            self.counter = 0
            self.best_model_state = deepcopy(model.state_dict())
            print(f"Loss improved to {normal_train_loss:.6f} - Model saved")
            return False
        else:
            self.counter += 1
            print(f"No improvement - Counter: {self.counter}/{self.patience}")
          
            if self.counter >= self.patience:
                print(f"Early stopping triggered!")
                print(f"Best loss: {self.best_train_loss:.6f}")
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
    def __init__(self, patience=2):
        self.patience = patience
        self.best_train_loss = float('inf')
        self.track_test_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, normal_train_loss, all_train_loss, model):
        if (normal_train_loss < self.best_train_loss):
            self.best_train_loss = normal_train_loss
            self.track_test_loss = all_train_loss
            self.counter = 0
            self.counter_test_loss = 0 
            self.best_model_state = deepcopy(model.state_dict())
            print(f"Loss improved to {normal_train_loss:.6f} - Model saved")
            return False
        else:
            self.counter += 1
            print(f"No improvement - Counter: {self.counter}/{self.patience}")
          
            if (self.counter >= self.patience) & (all_train_loss < self.track_test_loss):
                print(f"Early stopping triggered!")
                print(f"Best loss: {self.best_train_loss:.6f}")
                return True
        return False