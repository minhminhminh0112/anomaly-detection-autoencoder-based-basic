#PREPARE THE DATASET in a one hot encoded dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from typing import Union, Literal, Optional
import os
from dotenv import load_dotenv
load_dotenv('.env.local')

class BaseData:
    '''Base class for data preprocessing with common functionality for VehicleData and SyntheticData'''
    
    def __init__(self):
        # These will be set by child classes
        self.df = None
        self.X_raw = None
        self.y = None
        self.ordered_cols = None
        
    def _categorize_columns(self):
        '''Categorize columns into different types (date, boolean, numerical, categorical)'''
        if self.X_raw is None:
            raise ValueError('X_raw must be set before categorizing columns')
            
        self.date_cols = self.X_raw.select_dtypes(include=['datetime']).columns.tolist()
        self.bool_cols = self.X_raw.columns[self.X_raw.columns.str.contains('FLAG')].tolist()
        numerical_cols = self.X_raw.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
        id_cols = self.X_raw.columns[self.X_raw.columns.str.contains('ID') & ~self.X_raw.columns.isin(self.bool_cols)].tolist() #'VOTERID_FLAG'
        self.num_cols = [i for i in numerical_cols if i not in id_cols and i not in self.bool_cols]
        categorical_cols = self.X_raw.select_dtypes(include=['object']).columns.tolist()
        self.cat_cols = categorical_cols + id_cols
        self.ordered_cols = self.X_raw.columns.tolist()
    
    def transform(self, X: pd.DataFrame, scaler_type: Literal['standard', 'minmax'] = 'standard',
                  fitted_scaler=None, fitted_ohe=None):
        '''Transform data using specified scaler'''
        valid_scalers = ['standard', 'minmax']
        if scaler_type.lower() not in valid_scalers:
            raise ValueError(f'scaler_type must be one of {valid_scalers}, got {scaler_type}')

        if scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        return Transform(self, X=X, scaler=scaler, fitted_scaler=fitted_scaler, fitted_ohe=fitted_ohe)
    
    def get_X_train(self, array_format=True, scaler_type: Literal['standard', 'minmax'] = 'standard'):
        '''Get transformed training data'''
        return self.transform(X=self.X_raw, scaler_type=scaler_type).transform_input(array_format=array_format)

    def get_ohe_dict(self):
        '''Get one-hot encoding feature index mapping as a dictionary'''
        return self.transform(X=self.X_raw).get_OHE_columns_index()

class VehicleData(BaseData):
    def __init__(self, train_mode: bool = True):
        super().__init__()
        self.df = self.import_data(train=train_mode)
        self.df['DISBURSAL_DATE'] = pd.to_datetime(self.df['DISBURSAL_DATE'], format='%d-%m-%Y')
        self.df['DATE_OF_BIRTH'] = pd.to_datetime(self.df['DATE_OF_BIRTH'], format='%d-%m-%Y')
        self.df['AGE_DISBURSMENT'] = (self.df['DISBURSAL_DATE'] - self.df['DATE_OF_BIRTH']).dt.days // 365
        self.df['CREDIT_HISTORY_LENGTH'] = self.transform_yrs_mon('CREDIT_HISTORY_LENGTH')
        self.df.fillna({'EMPLOYMENT_TYPE': 'Not provided'}, inplace=True)
        self.df_wo_filter = self.df.drop([ 'DISBURSAL_DATE','UNIQUEID', 'CURRENT_PINCODE_ID', 'SUPPLIER_ID', 'EMPLOYEE_CODE_ID', 'BRANCH_ID', 'DATE_OF_BIRTH', 'PERFORM_CNS_SCORE', 'STATE_ID', 'MANUFACTURER_ID', 'MOBILENO_AVL_FLAG',
                                          'PRI_SANCTIONED_AMOUNT', 'SEC_SANCTIONED_AMOUNT', 'SEC_DISBURSED_AMOUNT', 'AVERAGE_ACCT_AGE'], axis = 1) # high correlation
        self.X_raw = self.df_wo_filter.drop(['LOAN_DEFAULT'], axis = 1)
        self.y = self.df_wo_filter['LOAN_DEFAULT']
        
        # Categorize columns using parent class method
        self._categorize_columns()

    def import_data(self, train: bool):
        path = os.getenv('DATASET_PATH', 'C:/Users/midon/.cache/kagglehub/datasets/avikpaul4u/vehicle-loan-default-prediction/versions/4/')
        if train:
            df = pd.read_csv(path + 'train.csv')
        else:
            df = pd.read_csv(path + 'test.csv')
        return df 
    
    def transform_yrs_mon(self, col_name: str):
        split_tab = self.df[col_name].str.split(' ', expand=True)
        split_tab.columns = ['years', 'months']
        split_tab['years'] = split_tab['years'].str.replace('yrs', '')
        split_tab['months'] = split_tab['months'].str.replace('mon', '')
        split_tab['years'] = split_tab['years'].astype(int)
        split_tab['months'] = split_tab['months'].astype(int)
        return split_tab['years'] * 12 + split_tab['months']

class SyntheticData(BaseData):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.df = data
        self.y = data['LOAN_DEFAULT']
        self.X_raw = self.df.drop(['LOAN_DEFAULT'], axis=1)
        self._categorize_columns()

class Transform:
    def __init__(self, data_class: BaseData, X: pd.DataFrame, scaler: Union[StandardScaler, MinMaxScaler],
                 fitted_scaler=None, fitted_ohe=None):
        self.data_class = data_class
        self.df = X
        self.sample_size = len(self.df)
        self.cat_cols = data_class.cat_cols
        self.num_cols = data_class.num_cols
        self.bool_cols = data_class.bool_cols
        self.date_cols = data_class.date_cols

        if self.num_cols or self.date_cols:
            self.num_df = pd.concat([self.df[self.num_cols],self.df[self.date_cols].astype('int64')], axis =1)
            # Use pre-fitted scaler if provided, otherwise fit a new one
            if fitted_scaler is not None:
                self.scaler = fitted_scaler
            else:
                self.scaler = scaler.fit(self.num_df)

        # Store pre-fitted OHE if provided
        self.fitted_ohe = fitted_ohe
        self.input_cols = self.get_OHEncoded_cols() + self.bool_cols + self.num_cols + self.date_cols

    def list_cat_cols(self):
        cat_dict = {}
        for col in self.cat_cols:
            unique_values = self.df[col].unique()
            cat_dict[col] = {'n':len(unique_values), 'values': unique_values}
        return cat_dict
    
    def get_min_max(self):
        min_values = self.num_df.min(axis=0)
        max_values = self.num_df.max(axis=0)
        return min_values, max_values

    def min_max_clipping(self, data:pd.DataFrame):
        min_values, max_values = self.get_min_max()
        for col in self.num_cols + self.date_cols:
            data[col] = data[col].clip(lower=min_values[col], upper=max_values[col])
        return data

    def scale(self):
        return self.scaler.transform(self.num_df)

    def reverse_scaler(self, data):
        return self.scaler.inverse_transform(data)
    
    def fit_OHE(self):
        # Use pre-fitted OHE if provided, otherwise fit a new one
        if self.fitted_ohe is not None:
            return self.fitted_ohe
        ohe = OneHotEncoder()
        ohe.fit(self.df[self.cat_cols])
        return ohe
    
    def get_OHE_columns_index(self):
        ohe = self.fit_OHE()
        ohe_features= ohe.feature_names_in_
        ohe_features_out = ohe.get_feature_names_out()
        features_dict = {}

        for i in range(len(ohe_features)):
            count = 0 
            start_ind = None
            for j in range(len(ohe_features_out)):
                if ohe_features[i] in ohe_features_out[j]:
                    if start_ind is None:
                        start_ind = j 
                    count += 1
            features_dict[ohe_features[i]] = {'start': start_ind, 'end': start_ind + count - 1 if start_ind is not None else None}
        return features_dict

    #Before encoding, should check how many categories are there to expect
    def OneHotEncoding(self, array_format=False):
        ohe = self.fit_OHE()
        cat_data = pd.DataFrame(ohe.transform(self.df[self.cat_cols]).toarray())
        cat_data.columns = ohe.get_feature_names_out()
        cat_data.columns = [str(col) for col in cat_data.columns]
        if array_format:
            return np.asarray(cat_data).astype('float32')
        else:
            return cat_data

    def transform_input(self, array_format=True):
        if array_format:
            cat_arr = self.OneHotEncoding(array_format=True)
            bool_arr = np.asarray(self.df[self.bool_cols])
            num_arr = self.scale() #include date cols
            input_arr = np.concatenate([cat_arr, bool_arr, num_arr], axis=1)
            return np.asarray(input_arr).astype('float32')
        else:
            #reset index to avoid misalignment with concatenation
            cat_df = self.OneHotEncoding(array_format=False).reset_index(drop=True)
            bool_df = self.df[self.bool_cols].reset_index(drop=True)
            num_df = pd.DataFrame(self.scale(), columns=self.num_cols + self.date_cols).reset_index(drop=True)
            input_df = pd.concat([cat_df, bool_df, num_df], axis=1)
            return input_df
            
    def get_OHEncoded_cols(self):
        return self.OneHotEncoding(array_format=False).columns.tolist()

    def solve_false_one_hot(self, row):
        while np.count_nonzero(row == 1) > 1:
            row[np.where(row == 1)[0][0]] = 0
        return row

    def restore_constraints(self, data):
        data_out = pd.DataFrame(columns=data.columns)

        encoded_col_index = 0
        for col in self.df.columns:
            if col in data.columns:
                data_out[col] = data[col].round() if self.df[col].dtype == 'int64' else data[col]
            else:
                num_values = len(self.df[col].unique())
                encoded_col_np = data[data.columns[encoded_col_index:encoded_col_index + num_values]].to_numpy()
                encoded_col_np = (encoded_col_np == encoded_col_np.max(axis=1)[:, None]).astype(int)
                for i, row in enumerate(encoded_col_np):
                    encoded_col_np[i] = self.solve_false_one_hot(row)
                data_out[data.columns[encoded_col_index:encoded_col_index + num_values]] = encoded_col_np
                encoded_col_index += num_values

        return data_out

    def reverse_one_hot(self, denorm_data: pd.DataFrame):
        reversed_df = pd.DataFrame(columns=self.df.columns)
        OHEncoded_cols = denorm_data.columns
        pred_denorm = pd.DataFrame(denorm_data, columns=OHEncoded_cols)

        for col in reversed_df.columns:
            if col in pred_denorm:
                reversed_df[col] = pred_denorm[col]
            else:
                col_OHEncoded = OHEncoded_cols[OHEncoded_cols.str.startswith(col)]
                reversed_df[col] = pred_denorm[col_OHEncoded].idxmax(axis=1).str.split('_').str[-1]
        return reversed_df

    def transform_preds(self, raw_preds, min_max_clipping:bool = True):
        if isinstance(raw_preds, torch.Tensor):
            raw_preds = raw_preds.numpy()
        cat_dims = len(self.get_OHEncoded_cols())
        bool_dims = len(self.bool_cols)
        data = raw_preds.copy()
        data[:, (cat_dims + bool_dims):] = self.reverse_scaler(data[:, (cat_dims + bool_dims):])
        data = pd.DataFrame(data, columns=self.input_cols)
        if min_max_clipping:
            data = self.min_max_clipping(data)
        restored_constraints = self.restore_constraints(data)
        reversed_OHE = self.reverse_one_hot(restored_constraints)
        reversed_OHE[self.date_cols] = reversed_OHE[self.date_cols].apply(pd.to_datetime, unit= 'ns')
        for col in self.date_cols:
            reversed_OHE[col] = pd.to_datetime(reversed_OHE[col], unit='ns').dt.strftime('%Y-%m-%d')
        return reversed_OHE

    def get_metadata(self):
        input_metadata = {}

        input_col_indices = {col: idx for idx, col in enumerate(self.input_cols)}

        for i in self.df.columns:
            start_index_exist = True
            n_values = 0
            for j in self.input_cols:
                if j.startswith(i) and start_index_exist:
                    start_index = input_col_indices[j]
                    input_metadata[i] = {'start_index': start_index}
                    start_index_exist = False
                if j.startswith(i):
                    n_values += 1

            input_metadata[i]['n_values'] = n_values
        return input_metadata
