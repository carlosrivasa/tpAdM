import mlflow
import pandas as pd
import boto3
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Global components
model = None
encoders = {}
scaler = None
numeric_columns = []
elo_df = None
results_df = None


def load_components():
    """Load all ML components (model, encoders, scaler) and data from MLflow and S3"""
    global model, encoders, scaler, numeric_columns, elo_df, results_df
    
    try:
        # Load model
        model = mlflow.sklearn.load_model("models:/fifa_win_nowin_match_predict/latest") 
        
        # Load encoders
        encoder_cols = ['home_team', 'away_team', 'tournament']
        encoders = {
            col: mlflow.sklearn.load_model(f"models:/fifa_win_nowin_match_predict_{col}_encoder/latest")
            for col in encoder_cols
        }
                
        # Load scaler
        scaler = mlflow.sklearn.load_model("models:/fifa_win_nowin_match_predict_standard_scaler/latest")
        
        # Load numeric columns from the scaler's run
        run_id = mlflow.tracking.MlflowClient().get_latest_versions(
            "fifa_win_nowin_match_predict_standard_scaler"
        )[0].run_id
        
        # Load numeric columns
        numeric_columns = mlflow.artifacts.load_dict(
            f"runs:/{run_id}/scalers/numeric_columns.json"
        ).get("numeric_columns", [])
        
        # Load ELO ratings
        elo_df = get_elo_ratings()
        
        # Load results data
        results_df = get_results_data()
        
        logger.info("All components loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading components: {str(e)}")
        return False


def get_elo_ratings():
    """Load ELO ratings data from S3"""
    s3 = boto3.client('s3',
                     endpoint_url='http://s3:9000',
                     aws_access_key_id='minio',
                     aws_secret_access_key='minio123')
    
    obj = s3.get_object(Bucket='data', Key='processed/elo_ratings.csv')
    return pd.read_csv(BytesIO(obj['Body'].read()))


def get_elo_rating_diff(home_team, away_team, match_date):
    """Get ELO rating difference between two teams for a given date"""
    # Convert match_date to pandas Timestamp if it's not already
    match_date = pd.to_datetime(match_date)
    
    # Filter first, then convert only the relevant dates
    filtered = elo_df[
        (elo_df['home_team'] == home_team) & 
        (elo_df['away_team'] == away_team)
    ].copy()  # Create a copy of the filtered data
    
    if not filtered.empty:
        # Convert only the dates we need to compare
        filtered.loc[:, 'date'] = pd.to_datetime(filtered['date'])
        recent_match = filtered[filtered['date'] <= match_date].sort_values('date', ascending=False).head(1)
        
        if not recent_match.empty:
            logger.info(f"ELO rating diff for {home_team} vs {away_team}: {recent_match['rating_diff'].iloc[0]}")
            return recent_match['rating_diff'].iloc[0]
    
    logger.warning(f"No ELO rating found for {home_team} vs {away_team} before {match_date}")
    return 0.0


def get_results_data():
    """Load FIFA results data from S3"""
    s3 = boto3.client('s3',
                     endpoint_url='http://s3:9000',
                     aws_access_key_id='minio',
                     aws_secret_access_key='minio123')
    obj = s3.get_object(Bucket='data', Key='datasets/results.csv')
    return pd.read_csv(BytesIO(obj['Body'].read()))
