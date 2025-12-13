from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator
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
    
    @field_validator('home_team', 'away_team')
    @classmethod
    def validate_team(cls, v):
        valid_teams = set(utils.results_df['home_team'].unique())
        if v not in valid_teams:
            raise ValueError(f'Team "{v}" not found in valid teams')
        return v
    
    @field_validator('tournament')
    @classmethod
    def validate_tournament(cls, v):
        valid_tournaments = {
            'FIFA World Cup qualification',
            'UEFA Euro qualification',
            'UEFA Euro',
            'FIFA World Cup',
            'Copa América',
            'UEFA Nations League'
        }
        if v not in valid_tournaments:
            raise ValueError(f'Tournament "{v}" must be one of: {", ".join(valid_tournaments)}')
        return v
    
    @model_validator(mode='after')
    def validate_different_teams(self):
        if self.home_team == self.away_team:
            raise ValueError('home_team and away_team must be different')
        return self


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
