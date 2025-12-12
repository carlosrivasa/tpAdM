from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from datetime import date
import logging
import boto3
from io import BytesIO
    
# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FIFA Match Prediction API")

# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Ajusta según tu configuración
MODEL_NAME = "win_nowin_fifa_match_predict"

# Initialize components
model = None
encoders = {}
scaler = None
numeric_columns = []
elo_df = None

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    tournament: str
    neutral: bool
    date: date

def load_components():
    global model, encoders, scaler, numeric_columns, elo_df
    
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
        
        logger.info("All components loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading components: {str(e)}")
        return False

# Cargar el modelo al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if not load_components():
        raise RuntimeError("Failed to load one or more components")

@app.get("/")
def read_root():
    return {"message": "FIFA Match Prediction API"}

# Ruta de predicción
@app.post("/predict")
async def predict(match: MatchRequest):
    try:
        # Calculate ELO ratings
        rating_diff = get_elo_rating_diff(
            match.home_team, match.away_team, match.date
        )
        
        try:
            # Prepare features
            features = {
                'neutral': [float(match.neutral)],
                'year': [float(match.date.year)],
                'month': [float(match.date.month)],
                'dayofweek': [float(match.date.weekday())],
                'rating_diff': [float(rating_diff)],
                'home_team': [match.home_team],
                'away_team': [match.away_team],
                'tournament': [match.tournament],
            }
            logger.info(f"Prepared features: {features}")
            # Create DataFrame
            df = pd.DataFrame(features)
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid Request")

        # Apply target encoding with better error handling
        for col in ['home_team', 'away_team', 'tournament']:
            df[col] = encoders[col].transform(df[col])
        
        # Scale numeric features
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        
        predictions = model.predict(df)
        predictions_proba = model.predict_proba(df)

        return {
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_team_win": bool(predictions[0]),
            "probability_home_win": float(predictions_proba[0][1]),
            "probability_nome_nowin": float(predictions_proba[0][0])
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para verificar la salud del modelo
@app.get("/health")
async def health_check():
    try:
        # Verificar que el modelo esté cargado
        if 'model' not in globals():
            raise Exception("Modelo no cargado")
        return {"status": "healthy", "model": MODEL_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
def get_elo_ratings():
    s3 = boto3.client('s3',
                     endpoint_url='http://s3:9000',
                     aws_access_key_id='minio',
                     aws_secret_access_key='minio123')
    
    obj = s3.get_object(Bucket='data', Key='processed/elo_ratings.csv')
    return pd.read_csv(BytesIO(obj['Body'].read()))

def get_elo_rating_diff(home_team, away_team, match_date):
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