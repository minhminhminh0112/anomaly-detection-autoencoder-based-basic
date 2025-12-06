import numpy as np 
import torch.nn as nn
import torch 
from typing import Union
from torch import Tensor

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

def get_error_score(model:nn.Module, X:np.ndarray, n_binary_cols:int) -> np.ndarray:
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    with torch.no_grad():
        recon = model(X)
    recon_err = X - recon
    error_score = recon_err.detach().numpy().copy()
    num_error_score = 1-np.exp(-(X[:,n_binary_cols:] - recon[:,n_binary_cols:])**2)
    error_score[:,n_binary_cols:] = num_error_score
    error_score[:,:n_binary_cols] = np.abs(error_score[:,:n_binary_cols])
    return error_score.sum(axis=1)

def get_error_score_prediction_train_test(X_train:np.ndarray, X_test:np.ndarray, top_n:int, model: nn.Module, n_binary_cols: int):
    error_score_train = get_error_score(model,X_train,n_binary_cols)
    error_score_test = get_error_score(model,X_test,n_binary_cols)
    pred_labels_train = get_top_n_prediction(error_score_train, top_n)
    threshold_value = np.sort(error_score_train)[-top_n]
    pred_labels_test = get_threshold_prediction(error_score_test, threshold_value)
    return pred_labels_train, pred_labels_test

