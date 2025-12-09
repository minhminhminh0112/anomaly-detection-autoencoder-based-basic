import os 
from dotenv import load_dotenv
load_dotenv('C:/Users/midon/Documents/anomaly-detection-autoencoder-based-basic/.env.local')
PROJECT_PATH = os.getenv('PROJECT_PATH')
os.chdir(PROJECT_PATH)
import sys
sys.path.append(PROJECT_PATH)

from preprocessing.preprocessing import * 
from sklearn.model_selection import train_test_split
import pickle
# try to restore the old and split X_train before using transform 
path = 'synthetic_data/data'
synth = SyntheticData(path + '/synthetic_data.csv')
new_features_data = add_new_features(synth.X_raw)
new_features_data['PERFORM_CNS_SCORE_DESCRIPTION'] = regroup_cat(new_features_data.PERFORM_CNS_SCORE_DESCRIPTION)
columns_high_corr_drop = ['ASSET_COST','CREDIT_HISTORY_LENGTH', 'AADHAR_FLAG','PRI_ACTIVE_ACCTS','SEC_ACTIVE_ACCTS',
                              'PRI_NO_OF_ACCTS','SEC_NO_OF_ACCTS','PRI_DISBURSED_AMOUNT','SEC_DISBURSED_AMOUNT',
                              'PRI_SANCTIONED_AMOUNT','SEC_SANCTIONED_AMOUNT','PRIMARY_INSTAL_AMT', 'AVERAGE_ACCT_AGE', 
                              'EMPLOYMENT_TYPE','DRIVING_FLAG', 'PAN_FLAG', 'PASSPORT_FLAG', 'SEC_OVERDUE_ACCTS', 'PRI_OVERDUE_ACCTS',
                              'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS']
new_data = new_features_data.drop(columns = columns_high_corr_drop)

X_train, X_test, y_train, y_test = train_test_split(
        new_data, synth.y, test_size=0.2, random_state=42)

final_data = type(synth).__new__(type(synth))
final_data.X_raw = X_train 
final_data.y = y_train
final_data.log_transform = False
final_data._categorize_columns() 
transformer_train = final_data.transform(final_data.X_raw) 
# SAVE
with open(path + '/transformer.pkl', 'wb') as f:
        pickle.dump(transformer_train, f)

data_dict = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}

with open(path + '/train_test_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)