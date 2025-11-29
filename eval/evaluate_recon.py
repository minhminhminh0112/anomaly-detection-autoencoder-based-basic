from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np 
from typing import Union

def evaluate_metrics(real_labels, pred_labels):
    f1 = f1_score(real_labels, pred_labels)
    precision = precision_score(real_labels, pred_labels)
    recall = recall_score(real_labels, pred_labels)
    accuracy = accuracy_score(real_labels, pred_labels)
    print(f"F1 Score: {round(f1, 4)}, Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, Accuracy: {round(accuracy, 4)}")
    return f1, precision, recall, accuracy

def confusion_matrix_metrics(real_labels, pred_labels):
    tn, fp, fn, tp = confusion_matrix(y_true = real_labels, y_pred = pred_labels).ravel().tolist()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    return tn, fp, fn, tp

def wasserstein_similarity(real_data:Union[pd.Series,np.ndarray], synthetic_data:Union[pd.Series,np.ndarray])-> float:
    from scipy.stats import wasserstein_distance
    evaluation_points = get_eval_points(real_data, synthetic_data)
    normalized_diff = wasserstein_distance(real_data, synthetic_data)/evaluation_points.mean()
    return round(1 - normalized_diff,2)

def get_eval_points(real_data:Union[pd.Series,np.ndarray], synthetic_data:Union[pd.Series,np.ndarray]):
    # becase the points are different from real and synthetic data
    combined_data = np.concatenate([real_data, synthetic_data])
    evaluation_points = np.sort(np.unique(combined_data))
    return evaluation_points

def evaluate_reconstructed(real_data:pd.DataFrame, rec_df:pd.DataFrame):
    from sdmetrics.reports.single_table import QualityReport
    from sdv.metadata import Metadata
    metadata = Metadata.detect_from_dataframe(
        data=real_data,
        table_name='df')
    metadata_dict= metadata.to_dict()['tables']['df']
    report = QualityReport()
    report.generate(real_data = real_data, synthetic_data= rec_df, metadata = metadata_dict)
    return report

def evaluate_table(real_data:pd.DataFrame, synthetic_data:pd.DataFrame, num_cols:list) -> pd.DataFrame:
    from scipy.spatial import distance 
    from sklearn.metrics import mean_absolute_error,r2_score
    report = evaluate_reconstructed(real_data = real_data, rec_df = synthetic_data)
    col_shapes_df = report.get_details("Column Shapes")
    wasserstein_score = {}
    for col in num_cols:
        score = wasserstein_similarity(real_data = real_data[col], synthetic_data= synthetic_data[col])
        wasserstein_score[col] = score
    col_shapes_df = report.get_details("Column Shapes")
    # col_shapes_df['bad_shape'] = [1 if shape < 0.7 else 0 for shape in col_shapes_df['Score']]
    wasserstein_df = pd.DataFrame.from_dict(wasserstein_score, orient='index').reset_index()
    wasserstein_df.columns = ['Column', 'Wasserstein Similarity']
    col_shapes_df = col_shapes_df.merge(wasserstein_df, on='Column', how='left')
    col_shapes_df.columns = ['Column','SDV Metric', 'Column Shape Score', 'Wasserstein Similarity']
    js_dist = distance.jensenshannon(real_data[num_cols], synthetic_data[num_cols], base=2,axis=0) #base of 2 so that the expected range is (0,1)
    js_similarity = 1- js_dist**2 #js_dist**2 is js divergence and because its base is 2 so that the expected range is (0,1)
    js_simi_dict = {}
    mae_dict = {}
    r2_dict = {}
    # mse_dict = {}
    for col in num_cols:
        js_simi_dict[col] = js_similarity[num_cols.index(col)].round(4)
        mae = mean_absolute_error(real_data[col], synthetic_data[col])
        mae_dict[col] = round(mae,2)
        # mse = mean_squared_error(vehicle.X_raw[col], rec_df[col])
        # mse_dict[col] = mse
        r2 = r2_score(real_data[col], synthetic_data[col])
        r2_dict[col] = round(r2,4)

    col_shapes_df['Jensen-Shannon Similarity'] = col_shapes_df['Column'].map(js_simi_dict)
    col_shapes_df['R2'] = col_shapes_df['Column'].map(r2_dict)
    col_shapes_df['MAE'] = col_shapes_df['Column'].map(mae_dict)
    return col_shapes_df