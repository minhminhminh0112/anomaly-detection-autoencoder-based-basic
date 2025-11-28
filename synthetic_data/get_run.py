import mlflow
import os
import sys
import pickle
import pandas as pd 
import numpy as np
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)
import torch
from synthetic_data.vae import VariationalAutoencoder
from mlflow.tracking import MlflowClient

def get_finished_runs(experiment_id:str):
    """
    Get all finished runs in a specific MLflow experiment.
    """
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string="status = 'FINISHED'")
    finished_runs = [run.info.run_id for run in runs]
    print(f"Finished runs in experiment {experiment_id}: {finished_runs}")
    return finished_runs

def get_hyperparams_from_run(run_id: str) -> dict:
    """
    Retrieve parameters from a given MLflow run ID.
    """
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    run = mlflow.get_run(run_id)
    params = run.data.params
    return params

def get_weights_from_run(run_id: str):
    """
    Retrieve model weights from a given MLflow run ID.
    """
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    run = mlflow.get_run(run_id)
    # Try weights.pth, then best_weights.pth, then final_weights.pth
    artifact_dir = os.path.join(mlruns_dir, run.info.experiment_id, run.info.run_id, "artifacts")
    candidates = ["weights.pth", "best_weights.pth", "final_weights.pth"]
    for fname in candidates:
        artifact_path = os.path.join(artifact_dir, fname)
        if os.path.exists(artifact_path):
            return torch.load(artifact_path)
    raise FileNotFoundError(f"None of the weights files found in {artifact_dir}. Tried: {', '.join(candidates)}")

def get_test_data_from_run(run_id: str):
    client = MlflowClient()
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    run = mlflow.get_run(run_id)
    finished_runs = get_finished_runs(run.info.experiment_id)
    exp_run = mlflow.get_run(finished_runs[0])
    experiment = mlflow.get_experiment(exp_run.info.experiment_id)
    path = os.path.join("saved_models", experiment.name)
    X_test = np.load(os.path.join(path, "X_test.npy"))
    y_test = np.load(os.path.join(path, "y_test.npy"))
    return X_test, y_test

def get_model_from_run(run_id: str, input_dims_manual:int = None):
    """
    Retrieve model architecture from a given MLflow run ID.
    """
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    model_uri = f"runs:/{run_id}/model"
    params = get_hyperparams_from_run(run_id)

    latent_dims = int(params["latent_dims"])
    layers_encoder = np.fromstring(params['layers_encoder'].strip('[]'), dtype=int, sep=' ') 
    layers_decoder = np.fromstring(params['layers_decoder'].strip('[]'), dtype=int, sep=' ') 
    input_dims = int(params["input_dims"])
    try:
        state_dict = get_weights_from_run(run_id)
        cat_dims_from_weights = state_dict['decoder.cat_output_layer.weight'].shape[0]
        print(f"Inferred cat_dims from saved weights: {cat_dims_from_weights}")
    except:
        cat_dims_from_weights = int(params["cat_dims"])
        print(f"Using cat_dims from parameters: {cat_dims_from_weights}")

    model = VariationalAutoencoder(
        latent_dims=latent_dims,
        layers_encoder=layers_encoder,
        layers_decoder=layers_decoder,
        input_dims=input_dims,
        latent_dist=params.get("latent_dist", "normal"),
        pre_cont_layer=int(params.get("pre_cont_layer", 16)),
        pre_cat_layer=int(params.get("pre_cat_layer", 32)),
        cat_dims=cat_dims_from_weights  # Use inferred dimensions
    )
    return model

def get_reconstructed_data_from_run(run_id: str) -> np.array:
    """
    Retrieve reconstructed data from a given MLflow run ID.
    """
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    state_dict = get_weights_from_run(run_id)
    model = get_model_from_run(run_id)
    params = get_hyperparams_from_run(run_id)
    scaler_type = params.get("scaler_type", "standard")
    preprocessor = get_preprocessor_from_experiment(run_id)
    X_train = preprocessor.get_X_train(array_format=True, scaler_type=scaler_type)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        cat_output,cont_output, z = model(X_tensor)
        reconstructed = torch.cat([cat_output, cont_output], dim=1)
        reconstructed, z = reconstructed.detach().numpy(), z.detach().numpy()
        return reconstructed,z

def get_all_metrics_from_run(run_id: str) -> dict:
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    run = mlflow.get_run(run_id)
    return run.data.metrics

def get_evaluation_metrics_from_run(run_id: str) -> dict:
    all_metrics = get_all_metrics_from_run(run_id)
    evaluation_metrics = {
        key: value for key, value in all_metrics.items() 
        if any(keyword in key for keyword in [
            'avg_', 'f1_score', 'precision', 'recall', 'accuracy', 'tn', 'fp', 'fn', 'tp'
        ])
    }
    
    return evaluation_metrics

def get_last_loss_from_run(run_id: str, metric:str = None):
    """
    Retrieve the last logged value of a specified metric from a given MLflow run ID.
    """
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    run = mlflow.get_run(run_id)
    if metric:
        return run.data.metrics.get(metric)
    else:
        return run.data.metrics

def get_data_from_run(run_id: str):
    client = MlflowClient()
    artifact_path = client.download_artifacts(run_id, "data_preprocessor.pkl")
    with open(artifact_path, "rb") as f:
        vehicle = pickle.load(f)
    return vehicle

def get_transformer_from_experiment(run_id:str):
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    client = MlflowClient()
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    experiment_name = experiment.name
    path = os.path.join("saved_models", experiment_name, "transformer.pkl")
    with open(path, "rb") as f:
        transformer = pickle.load(f)
    return transformer

def get_recon_transformed_data(run_id:str) -> pd.DataFrame:
    reconstructed,z = get_reconstructed_data_from_run(run_id)
    transformer = get_transformer_from_experiment(run_id)
    rec_df = transformer.transform_preds(reconstructed)
    rec_df = rec_df.astype(transformer.df.dtypes)
    return rec_df

# Log artifact to the existing run
def log_evaluation_df(run_id:str, col_shapes_df: pd.DataFrame, real_labels, pred_labels):    
    with mlflow.start_run(run_id=run_id):
        print(f"Logging artifacts to existing run: {run_id}")
        
        csv_content = col_shapes_df.to_csv(index=False)
        mlflow.log_text(csv_content, "evaluation_df.csv")
        print("col_shapes_df.csv logged as artifact successfully!")

        # Optionally log some summary metrics
        mlflow.log_metric("avg_column_shape_score", col_shapes_df['Column Shape Score'].mean())
        mlflow.log_metric("avg_wasserstein_similarity", col_shapes_df['Wasserstein Similarity'].mean())
        mlflow.log_metric("avg_js_similarity", col_shapes_df['Jensen-Shannon Similarity'].replace(-np.inf, np.nan).mean())
        mlflow.log_metric("avg_r2_score", col_shapes_df['R2'].mean())
        mlflow.log_metric("avg_mae", col_shapes_df['MAE'].mean())
        # avoid circular import
        from eval.evaluate_recon import evaluate_metrics, confusion_matrix_metrics
        f1, precision, recall, accuracy = evaluate_metrics(real_labels = real_labels, pred_labels = pred_labels)
        tn, fp, fn, tp = confusion_matrix_metrics(real_labels = real_labels, pred_labels = pred_labels)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)
        print(f"Summary metrics also logged to run: {run_id}")

        
def get_eval_table(run_id: str) -> pd.DataFrame:
    """
    Retrieve the evaluation table for a given MLflow run ID.
    """
    mlruns_dir = "mlruns"
    mlflow.set_tracking_uri(mlruns_dir)
    client = MlflowClient()
    run = client.get_run(run_id)
    
    # Read directly from the mlruns directory
    direct_path = os.path.join(mlruns_dir, run.info.experiment_id, run.info.run_id, "artifacts", "evaluation_df.csv")
    
    if os.path.exists(direct_path):
        eval_table = pd.read_csv(direct_path)
        return eval_table
    else:
        raise FileNotFoundError(f"Could not find evaluation_df.csv in artifacts for run {run_id} at path: {direct_path}")

