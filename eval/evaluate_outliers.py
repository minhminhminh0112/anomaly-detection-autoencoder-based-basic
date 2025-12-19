import numpy as np 
import torch.nn as nn
import torch 
from typing import Union
from torch import Tensor
import pandas as pd

def compute_per_sample_loss(
    model: nn.Module,
    X: Union[np.ndarray,Tensor],
    n_binary_cols: int,
    num_weight: float,
    bool_weight: float,
    num_criterion = None,
    bool_criterion = None, 
    training = True 
) -> Union[np.ndarray,Tensor]:
    """
    Return an array with the reconstruction loss for each sample in `X`.
    The loss is a weighted sum of a MSE for the numerical columns and a
    BCE for the binary columns.
    """
    if num_criterion is None:
        num_criterion = nn.MSELoss(reduction='none')
    if bool_criterion is None:
        bool_criterion = nn.BCELoss(reduction='none')

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if training:
        recon = model(X)
    else: 
        with torch.no_grad():
            recon = model(X)
    
    num_loss = num_criterion(recon[:, n_binary_cols:], X[:, n_binary_cols:]).sum(dim=1)
    bool_loss = bool_criterion(recon[:, :n_binary_cols], X[:, :n_binary_cols]).sum(dim=1)

    per_sample_loss = num_weight * num_loss + bool_weight * bool_loss

    if isinstance(X, np.ndarray):
        per_sample_loss = per_sample_loss.detach().numpy()

    if training:
        return num_loss, bool_loss, per_sample_loss
    else:
        return per_sample_loss
    
def compute_denoising_per_sample_loss(
    model: nn.Module,
    X: Union[np.ndarray,Tensor],
    n_binary_cols: int,
    num_weight: float,
    bool_weight: float,
    alpha:float,
    mask: np.ndarray,
    num_criterion = None,
    bool_criterion = None, 
    training = True
) -> Union[np.ndarray,Tensor]:
    """
    Return an array with the reconstruction loss for each sample in `X`.
    The loss is a weighted sum of a MSE for the numerical columns and a
    BCE for the binary columns.
    """
    if num_criterion is None:
        num_criterion = nn.MSELoss(reduction='none')
    if bool_criterion is None:
        bool_criterion = nn.BCELoss(reduction='none')

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()

    recon = model(X)
    m_bool = mask[:, :n_binary_cols]
    m_num = mask[:, n_binary_cols:]

    num_loss = num_criterion(recon[:, n_binary_cols:], X[:, n_binary_cols:])
    bool_loss = bool_criterion(recon[:, :n_binary_cols], X[:, :n_binary_cols])
    enhanced_bool_loss = alpha * m_bool * bool_loss + (1.0 - alpha) * (1.0 - m_bool) * bool_loss
    enhanced_num_loss  = alpha * m_num  * num_loss  + (1.0 - alpha) * (1.0 - m_num)  * num_loss

    per_sample_loss = num_weight * enhanced_num_loss.sum(dim=1) + bool_weight * enhanced_bool_loss.sum(dim=1)

    if isinstance(X, np.ndarray):
        per_sample_loss = per_sample_loss.cpu().numpy()

    if training:
        return num_loss, bool_loss, per_sample_loss
    else:
        return per_sample_loss

def get_top_n_prediction(per_sample_loss:np.ndarray, top_n: int):
    top_loss_idx = np.argsort(per_sample_loss)[-top_n:]
    pred_labels = np.zeros_like(per_sample_loss)
    pred_labels[top_loss_idx] = 1
    return pred_labels

def get_threshold_prediction(per_sample_loss:np.ndarray, threshold:float):
    pred_labels = np.zeros_like(per_sample_loss)
    pred_labels[per_sample_loss >= threshold] = 1
    return pred_labels

def get_loss_prediction_train_test(X_train:np.ndarray, X_test:np.ndarray, top_n:int, model: nn.Module,
    n_binary_cols: int,
    num_weight: float,
    bool_weight: float,
    num_criterion = None,
    bool_criterion = None):

    per_sample_loss_train = compute_per_sample_loss(model,X_train,n_binary_cols,num_weight,bool_weight, num_criterion, bool_criterion,training= False )
    per_sample_loss_test = compute_per_sample_loss(model,X_test,n_binary_cols,num_weight,bool_weight, num_criterion, bool_criterion, training=False)
    pred_labels_train = get_top_n_prediction(per_sample_loss_train, top_n)
    threshold_value = np.sort(per_sample_loss_train)[-top_n].item()
    pred_labels_test = get_threshold_prediction(per_sample_loss_test,threshold_value)
    return pred_labels_train, pred_labels_test

def get_error_score_prediction_train_test(X_train:np.ndarray, X_test:np.ndarray, top_n:int, model: nn.Module, attributes_info:dict, cat_cols:list, bool_cols:list, alpha:float =1.0 ):
    error_score_train = get_error_score(model= model, X=X_train, attributes_info=attributes_info, cat_cols=cat_cols, bool_cols=bool_cols ,alpha=alpha)
    error_score_train = error_score_train.mean(axis=1)
    error_score_test = get_error_score(model= model, X=X_test, attributes_info=attributes_info, cat_cols=cat_cols, bool_cols=bool_cols, alpha=alpha)
    error_score_test= error_score_test.mean(axis=1)
    pred_labels_train = get_top_n_prediction(error_score_train, top_n)
    threshold_value = np.sort(error_score_train)[-top_n]
    pred_labels_test = get_threshold_prediction(error_score_test, threshold_value)
    return pred_labels_train, pred_labels_test

def get_error_score(model:torch.nn.Module, X:np.array, attributes_info:dict, cat_cols:list, bool_cols:list, X_df:pd.DataFrame=None, alpha:float = 1.0) -> Union[np.ndarray, pd.DataFrame]:

    # X_t = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else X.float()
    X_t = torch.from_numpy(X)
    cat_set = set(cat_cols)
    bool_set = set(bool_cols)
    col_index_map = {c: i for i, c in enumerate(X_df.columns)} if X_df is not None else None

    model.eval()
    with torch.no_grad():
        recon = model(X_t)

    n_samples = X_t.shape[0]
    n_cols = len(X_df.columns) if X_df is not None else len(attributes_info)
    errs = torch.empty((n_samples, n_cols), dtype=recon.dtype)

    for i, (key, item) in enumerate(attributes_info.items()):
        col_idx = col_index_map[key] if col_index_map is not None else i
        start = item['start_index']
        end = start + item['n_values']

        if key in cat_set:
            vals = 1 - recon[:, start:end].max(dim=1).values
        elif key in bool_set:
            vals = (X_t[:, start] - recon[:, start]).abs()
        else:
            diff = X_t[:, start] - recon[:, start]
            vals = 1 - torch.exp(-alpha*(diff**2))
        errs[:, col_idx] = vals

    result = errs.detach().numpy()
    return result if X_df is None else pd.DataFrame(result, columns=X_df.columns)

def compute_percentile_ranks(errors:np.ndarray):
    ranks = np.argsort(np.argsort(errors)) + 1
    percentiles = (ranks / len(errors)) * 100
    
    return percentiles

# y_true = np.repeat(1,10)
# y_pred = np.arange(0,1,0.1)

# def soft_margin_loss(y_true, y_pred, epsilon=1e-15):
#     """
#     Element-wise loss using sigmoid, naturally bounded [0, 1]
#     """
#     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
#     margin = (2 * y_true - 1) * (2 * y_pred - 1)
    
#     loss = 1 / (1 + np.exp(margin))  
    
#     return loss

# loss = soft_margin_loss(y_true, y_pred)
# loss