from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from datetime import date
import logging
from utils import load_components, get_elo_rating_diff
import utils

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FIFA Match Prediction API")

# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "win_nowin_fifa_match_predict"


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    tournament: str
    neutral: bool
    date: date


# Cargar el modelo al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if not load_components():
        raise RuntimeError("Failed to load one or more components")


@app.get("/")
def read_root():
    return {"message": "FIFA Match Prediction API"}


@app.get("/teams")
def get_teams():
    try:
        teams = sorted(utils.results_df['home_team'].unique().tolist())

        return {"teams": teams}
    except Exception as e:
        logger.error(f"Error fetching teams: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
            df[col] = utils.encoders[col].transform(df[col])
        
        # Scale numeric features
        df[utils.numeric_columns] = utils.scaler.transform(df[utils.numeric_columns])
        
        predictions = utils.model.predict(df)
        predictions_proba = utils.model.predict_proba(df)

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
        if utils.model is None:
            raise Exception("Modelo no cargado")
        return {"status": "healthy", "model": MODEL_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
