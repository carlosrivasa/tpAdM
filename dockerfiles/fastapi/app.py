from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Any, Union
import mlflow
import pandas as pd
from datetime import date as date_type
import logging
from utils import load_components, get_elo_rating_diff
import utils

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="El Champion 🏆 / FIFA Match Prediction API ⭐⭐⭐")

# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "win_nowin_fifa_match_predict"


class MatchRequest(BaseModel):
    """Modelo para la solicitud de predicción de partido"""
    home_team: str = Field(
        ...,
        description="Nombre del equipo local, en inglés",
        example="Argentina"
    )
    away_team: str = Field(
        ...,
        description="Nombre del equipo visitante, en inglés",
        example="Brazil"
    )
    tournament: str = Field(
        ...,
        description="Torneo en el que se juega el partido, en inglés",
        example="FIFA World Cup"
    )
    neutral: bool = Field(
        ...,
        description="Indica si el partido se juega en el país del local",
        example=False
    )
    date: date_type = Field(
        ...,
        description="Fecha del partido en formato YYYY-MM-DD",
        example="2026-07-19"
    )
    
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
    return {"message": "🏆 El Champion ⭐⭐⭐"}


@app.get(
    "/teams",
    response_model=Dict[str, List[str]],
    summary="Obtener lista de equipos disponibles",
    description="Devuelve una lista de todos los equipos disponibles para usar en las predicciones",
    responses={
        200: {
            "description": "Lista de equipos obtenida exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "teams": ["Argentina", "Brazil", "Germany", "France", "Spain"]
                    }
                }
            }
        },
        500: {
            "description": "Error interno del servidor",
            "content": {
                "application/json": {
                    "example": {"detail": "Error al cargar la lista de equipos"}
                }
            }
        }
    }
)
def get_teams():
    try:
        teams = sorted(utils.results_df['home_team'].unique().tolist())
        return {"teams": teams}
    except Exception as e:
        logger.error(f"Error fetching teams: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al cargar la lista de equipos"
        )


@app.post(
    "/predict",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Predecir resultado de partido",
    description="Realiza la predicción del resultado de un partido entre dos equipos",
    responses={
        200: {
            "description": "Predicción exitosa",
            "content": {
                "application/json": {
                    "example": {
                        "home_team": "Argentina",
                        "away_team": "Brazil",
                        "home_team_win": True,
                        "probability_home_win": 0.78,
                        "probability_nome_nowin": 0.22
                    }
                }
            }
        },
        400: {
            "description": "Solicitud inválida",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid Request"}
                }
            }
        },
        422: {
            "description": "Error de validación de datos",
            "content": {
                "application/json": {
                    "example": {"detail": "Datos de entrada inválidos o faltantes"}
                }
            }
        },
        500: {
            "description": "Error interno del servidor",
            "content": {
                "application/json": {
                    "example": {"detail": "Error al procesar la predicción"}
                }
            }
        }
    }
)
async def predict(match_data: MatchRequest) -> Dict[str, Union[str, bool, float]]:
    try:
        # Calculate ELO ratings
        rating_diff = get_elo_rating_diff(
            match_data.home_team, match_data.away_team, match_data.date
        )
        
        try:
            # Prepare features
            features = {
                'neutral': [float(match_data.neutral)],
                'year': [float(match_data.date.year)],
                'month': [float(match_data.date.month)],
                'dayofweek': [float(match_data.date.weekday())],
                'rating_diff': [float(rating_diff)],
                'home_team': [match_data.home_team],
                'away_team': [match_data.away_team],
                'tournament': [match_data.tournament],
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
            "home_team": match_data.home_team,
            "away_team": match_data.away_team,
            "home_team_win": bool(predictions[0]),
            "probability_home_win": float(predictions_proba[0][1]),
            "probability_nome_nowin": float(predictions_proba[0][0])
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al procesar la predicción"
        )


# Ruta para verificar la salud del modelo
@app.get(
    "/health",
    response_model=Dict[str, str],
    summary="Verificar estado del servicio",
    description="Verifica que el servicio esté funcionando correctamente y el modelo esté cargado",
    responses={
        200: {
            "description": "Servicio funcionando correctamente",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model": "win_nowin_fifa_match_predict"
                    }
                }
            }
        },
        500: {
            "description": "Error en el servicio",
            "content": {
                "application/json": {
                    "example": {"detail": "Modelo no cargado"}
                }
            }
        }
    }
)
async def health_check():
    """Verifica el estado de salud del servicio.
    
    Returns:
        Dict: Estado del servicio y nombre del modelo cargado
    """
    try:
        if utils.model is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Modelo no cargado"
            )
        return {"status": "healthy", "model": MODEL_NAME}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error en el servicio"
        )
