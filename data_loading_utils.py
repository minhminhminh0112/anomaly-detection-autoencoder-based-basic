"""
Utility functions for loading and preprocessing vehicle data.
"""
import os 
from dotenv import load_dotenv
load_dotenv('.env.local')
PROJECT_PATH = os.getenv('PROJECT_PATH')
os.chdir(PROJECT_PATH)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from preprocessing import VehicleData, DataTransformer
import json

def load_final_data():
    vehicle_data = VehicleData(train_mode=True)
    y = vehicle_data.y
    featured_data = vehicle_data.X_raw[vehicle_data.num_cols].copy()

    # Feature engineering
    print("Applying feature engineering...")
    featured_data['LTA'] = (featured_data['DISBURSED_AMOUNT'] / featured_data['ASSET_COST'])
    featured_data['AGE_DISBURSMENT_LTV'] = featured_data['AGE_DISBURSMENT'] * featured_data['LTV']
    featured_data['DOWNPAYMENT'] = (featured_data['ASSET_COST'] - featured_data['DISBURSED_AMOUNT'])
    featured_data['SANCTION_GAP_PRI'] = featured_data['PRI_SANCTIONED_AMOUNT'] - featured_data['PRI_DISBURSED_AMOUNT']
    featured_data['SANCTION_GAP_SEC'] = featured_data['SEC_SANCTIONED_AMOUNT'] - featured_data['SEC_DISBURSED_AMOUNT']
    featured_data['DISB_SANC_RATIO'] = featured_data['DISBURSED_AMOUNT'] / (featured_data['PRI_SANCTIONED_AMOUNT'] + 0.1)
    featured_data['LTA_LTV'] = featured_data['LTA'] / featured_data['LTV'] * 100
    featured_data['TOTAL_EMI'] = featured_data['PRIMARY_INSTAL_AMT'] + featured_data['SEC_INSTAL_AMT']
    featured_data['EMI_DISBURSED_RATIO'] = featured_data['TOTAL_EMI'] / (featured_data['PRI_DISBURSED_AMOUNT']+ featured_data['SEC_DISBURSED_AMOUNT']+1.1)
    featured_data['PRI_EMI_DISBURSED_RATIO'] = featured_data['PRIMARY_INSTAL_AMT'] / (featured_data['PRI_DISBURSED_AMOUNT']+1.1)
    featured_data['SEC_EMI_DISBURSED_RATIO'] = featured_data['SEC_INSTAL_AMT'] / (featured_data['SEC_DISBURSED_AMOUNT']+1.1)

    featured_data['OUTSTANDING_BALANCE'] = featured_data['PRI_CURRENT_BALANCE'] + featured_data['SEC_CURRENT_BALANCE']
    featured_data['OUT_BALANCE_DISBURSED_RATIO'] = featured_data['OUTSTANDING_BALANCE'] / (featured_data['PRI_DISBURSED_AMOUNT']+ featured_data['SEC_DISBURSED_AMOUNT']+1.1)
    featured_data['OUT_BALANCE_SANCTIONED_RATIO'] = featured_data['OUTSTANDING_BALANCE'] / (featured_data['PRI_SANCTIONED_AMOUNT']+ featured_data['SEC_SANCTIONED_AMOUNT']+1.1)
    non_log_features = ['LTV', 'AGE_DISBURSMENT', 'IS_SALARIED', 'LTA']
    log_features = [col for col in featured_data.columns if col not in non_log_features]
    featured_log_data = np.log(featured_data[log_features]+1.1)
        
    yeojohnson_cols = ['PRI_CURRENT_BALANCE', 'SEC_CURRENT_BALANCE', 'OUTSTANDING_BALANCE',
                        'SANCTION_GAP_PRI', 'SANCTION_GAP_SEC', 'DISB_SANC_RATIO','OUT_BALANCE_DISBURSED_RATIO','OUT_BALANCE_SANCTIONED_RATIO']
    # Yeo-Johnson transformations
    print("Applying Yeo-Johnson transformations...")
    for col in yeojohnson_cols:
        yeojohnson = PowerTransformer(method='yeo-johnson')
        featured_log_data[col] = yeojohnson.fit_transform(featured_data[[col]])

    # Combine with boolean columns
    print("Adding boolean features and non_log features...")
    final_data = pd.concat([featured_log_data, featured_data[non_log_features], vehicle_data.X_raw[vehicle_data.bool_cols]], axis=1)
    final_data['SUM_FLAGS'] = final_data[vehicle_data.bool_cols].sum(axis=1)
    return final_data, vehicle_data.y

def load_and_preprocess_vehicle_data(test_size=0.2, random_state=42, scaler_type='standard', chosen_features_path:str=None):
    """
    Load vehicle data, apply feature engineering, transformations, and return train/val splits.

    Args:
        test_size: Proportion of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        scaler_type: Type of scaler to use - 'standard' or 'minmax' (default: 'standard')

    Returns:
        tuple: (X_train, X_val, y_train, y_val, transformer)
            - X_train: Training features (numpy array)
            - X_val: Validation features (numpy array)
            - y_train: Training labels (numpy array)
            - y_val: Validation labels (numpy array)
            - transformer: Fitted DataTransformer object for inverse transforms
    """
    final_data, vehicle_data_y = load_final_data()

    if chosen_features_path is not None:
        with open(chosen_features_path, 'r') as f:
            chosen_data = json.load(f)
        selected_features = chosen_data['selected_features']
        data = final_data[selected_features]
    else:
        data = final_data.copy()
    y = vehicle_data_y.values
    # Transform data using DataTransformer
    print(f"Applying {scaler_type} scaling...")
    transformer = DataTransformer.from_data(data, scaler_type=scaler_type)
    X_transformed = transformer.transform(data, array_format=True)

    print(f"Transformed data shape: {X_transformed.shape}")
    print(f"Fraud rate in data: {y.mean():.4f} ({y.sum()}/{len(y)})")

    # Split into train and validation sets
    print(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=test_size, random_state=random_state
    )

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Train fraud rate: {y_train.mean():.4f} ({y_train.sum()}/{len(y_train)})")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Val fraud rate: {y_val.mean():.4f} ({y_val.sum()}/{len(y_val)})")

    return X_train, X_val, y_train, y_val, transformer

