import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import mlflow
from mlflow.models.signature import infer_signature
import logging
from typing import Tuple, Dict, Any
import boto3
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_s3(bucket: str, x_key: str, y_key: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load training data from S3 bucket."""
    try:
        s3 = boto3.client('s3',
                         endpoint_url='http://s3:9000',
                         aws_access_key_id='minio',
                         aws_secret_access_key='minio123')
        
        # Load X_train
        x_obj = s3.get_object(Bucket=bucket, Key=x_key)
        X_train = pd.read_csv(BytesIO(x_obj['Body'].read()))
        
        # Load y_train
        y_obj = s3.get_object(Bucket=bucket, Key=y_key)
        y_train = pd.read_csv(BytesIO(y_obj['Body'].read()))['target'].values
        
        logger.info(f"Successfully loaded training data from s3://{bucket}/{x_key}")
        return X_train, y_train
    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise

def log_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Calculate and log metrics."""
    # Calculate metrics
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1": f1_score(y_true, y_pred),
        f"{prefix}roc_auc": roc_auc_score(y_true, y_pred),
    }
    
    # Log numeric metrics
    mlflow.log_metrics(metrics)
    
    # Log classification report as a text artifact
    report = classification_report(y_true, y_pred)
    mlflow.log_text(report, f"{prefix}classification_report.txt")
    
    return metrics

def train_test_model(X_train: pd.DataFrame, y_train: np.ndarray, 
                X_test: pd.DataFrame, y_test: np.ndarray,
                params: Dict[str, Any] = None, run_name: str = None) -> str:
    """
    Train model and evaluate on test set, returning MLflow run ID.
    """
    if params is None:
        params = {}
        
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(params)
        
        # Initialize and train model
        logger.info("Initializing and training RandomForest model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        test_metrics = log_metrics(y_test, y_pred, "test_")
        logger.info(f"Test metrics: {test_metrics}")
        log_metrics(y_test, y_pred, "test_")
        
        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="fifa_win_nowin_match_predict",
            signature=signature
        )
        
        return run.info.run_id

def setup_mlflow(experiment_name="fifa_match_hp_search_train_test"):
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "fifa-2026-win-nowin",
                "team": "mlops1-fiuba"
            }
        )
    else:
        experiment_id = experiment.experiment_id
        
    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id

def hyperparameter_search(
    X_train: pd.DataFrame, 
    y_train: np.ndarray,
    run_name: str = "hp_search"
) -> Tuple[Dict[str, Any], RandomForestClassifier]:
    """Perform hyperparameter search and return best parameters and model."""

    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [12, 15, 18],
        'min_samples_split': [6, 8, 10],
        'min_samples_leaf': [2, 3, 4],
        'criterion': ['gini', 'entropy']
    }

    best_model = None
    best_params = None
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"param_grid": str(param_grid)})

        model = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring='f1',
            cv=3,
            n_iter=30,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_params.update({
            'class_weight': 'balanced_subsample',
            'random_state': 42,
            'n_jobs': -1
        })

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_f1", search.best_score_)
        
    return best_params, best_model  # Devolvemos ambos

def main():
    """Main function to run the training and evaluation pipeline."""
    try:
        # Set up MLflow
        setup_mlflow()
        
        # Configuration
        bucket = "data"

        x_train_key = "processed/X_train_scaled.csv"
        y_train_key = "processed/y_train.csv"
        x_test_key = "processed/X_test_scaled.csv"
        y_test_key = "processed/y_test.csv"
        
        # Load training data
        logger.info("Loading training data from S3...")
        X_train_full, y_train_full = load_data_from_s3(bucket, x_train_key, y_train_key)

        # Load test data
        logger.info("Loading test data from S3...")
        X_test, y_test = load_data_from_s3(bucket, x_test_key, y_test_key)
        
        # Train models and return the best one
        logger.info("Starting hyperparameter and model search...")
        best_params, _ = hyperparameter_search(
            X_train=X_train_full,
            y_train=y_train_full,
            run_name="hp_search"
        )

        # Train the best model and evaluate
        run_id = train_test_model(
            X_train=X_train_full,
            y_train=y_train_full,
            X_test=X_test,
            y_test=y_test,
            params=best_params,
            run_name="train_test_run"
        )
        
        logger.info(f"Training and evaluation completed! MLflow Run ID: {run_id}")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()