import pandas as pd
import numpy as np
import category_encoders as ce
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import deque
from math import log
import logging
import boto3
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_YEAR = 1920
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client('s3',
                     endpoint_url='http://s3:9000',
                     aws_access_key_id='minio',
                     aws_secret_access_key='minio123')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

def save_df_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """
    Save a DataFrame to an S3 bucket as a CSV file.

    :param df: DataFrame to save
    :param bucket: S3 bucket name
    :param key: S3 object key (path + filename)
    """
    s3 = boto3.client('s3',
                     endpoint_url='http://s3:9000',
                     aws_access_key_id='minio',
                     aws_secret_access_key='minio123')
    
    # Convert DataFrame to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload to S3
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue()
    )
    logger.info(f"Successfully saved DataFrame to s3://{bucket}/{key}")

def save_text_to_s3(content: str, bucket: str, key: str) -> None:
    """
    Save text content to an S3 bucket.

    :param content: Text content to save
    :param bucket: S3 bucket name
    :param key: S3 object key (path + filename)
    """
    s3 = boto3.client('s3',
                     endpoint_url='http://s3:9000',
                     aws_access_key_id='minio',
                     aws_secret_access_key='minio123')
    
    # Upload text content to S3
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=content.encode('utf-8'),
        ContentType='text/plain'
    )
    logger.info(f"Successfully saved text to s3://{bucket}/{key}")
    
def preprocess_features(results: pd.DataFrame, min_year: int = MIN_YEAR) -> pd.DataFrame:
    """
    Preprocesa las características del dataset.

    :param dataset: Dataframe con el dataset
    :type dataset: pd.DataFrame
    :param min_year: Año mínimo para filtrar los datos
    :type min_year: int
    :returns: Dataset con las características preprocesadas
    :rtype: pd.DataFrame
    """

    # Convertir la columna de fecha a tipo datetime
    results["date"] = pd.to_datetime(results["date"])
    
    # Extraer características de la fecha
    results["year"] = results["date"].dt.year
    
     # Filtrar por fecha minima
    results = results[results["year"] >= min_year].copy()
     # Filtrar partidos oficiales
    results = results[results["tournament"] != 'Friendly'].reset_index(drop=True)
    
    results["month"] = results["date"].dt.month
    results["dayofweek"] = results["date"].dt.dayofweek
    results["is_weekend"] = (results["dayofweek"] >= 5).astype(int)

    results["neutral"] = results["neutral"].astype(int)
    
    # Calcular diferencia de goles y total de goles
    results["goal_diff"] = results["home_score"] - results["away_score"]
    results["total_score"] = results["home_score"] + results["away_score"]
    
    # Crear variable objetivo (win/nowin)
    results["result"] = np.where(results["home_score"] > results["away_score"], "win", "nowin")   
    results["target"] = np.where(results["goal_diff"] > 0, 1, 0)

    # Calcular ranking ELO
    elo_df = compute_elo_features(results)
    results_with_rating = results.merge(
        elo_df, on=["date", "home_team", "away_team"], how="left"
    )
    return results_with_rating


def select_features(
    data: pd.DataFrame,
    numerical_features: list,
    categorical_features: list,
    target_column: str
) -> pd.DataFrame:
    """
    Selecciona las características especificadas para el modelado.
    
    :param data: DataFrame con el dataset
    :type data: pd.DataFrame
    :param numerical_features: Lista de nombres de características numéricas a incluir
    :type numerical_features: list
    :param categorical_features: Lista de nombres de características categóricas a incluir
    :type categorical_features: list
    :param target_column: Nombre de la columna target
    :type target_column: str
    :return: DataFrame con las características seleccionadas
    :type: pd.DataFrame
    """
    # Combine all features we want to keep
    features_to_keep = numerical_features + categorical_features + [target_column]
    # Ensure all requested columns exist in the dataset
    features_to_keep = [f for f in features_to_keep if f in data.columns]
    
    return data[features_to_keep]


def make_dummies_variables(
    dataset: pd.DataFrame,
    categories_list: list,
    target_column: str = 'target'
    ) -> pd.DataFrame:
    """
    Converts categorical variables using target encoding.

    :param dataset: DataFrame with the dataset
    :type dataset: pd.DataFrame
    :param categories_list: List of categorical column names to encode
    :type categories_list: list
    :param target_column: Name of the target column for encoding
    :type target_column: str
    :returns: DataFrame with encoded columns
    :rtype: pd.DataFrame
    """

    # Copiar el dataset para evitar modificar el original
    encoded_data = dataset.copy()
    
    # Inicializar el target encoder
    encoder = ce.TargetEncoder(handle_missing='value', handle_unknown='value')
    
    # Aplicar target encoding a las columnas categóricas
    for col in categories_list:
        if col in encoded_data.columns:
            encoded_data[col] = encoder.fit_transform(encoded_data[col], encoded_data[target_column])
    
    # Log de las nuevas columnas
    content = "New columns after target encoding:\n" + "\n".join(encoded_data.columns.tolist())
    save_text_to_s3(content, "data", "output/log_columns_dummies.txt")
    
    return encoded_data


def split_dataset(
    dataset: pd.DataFrame,
    test_size: float,
    target_column: str,
    is_stratified: bool = True
    ) -> tuple:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.

    :param dataset: Dataframe con el dataset
    :type dataset: pd.DataFrame
    :param test_size: Proporción del set de prueba (entre 0 y 1)
    :type test_size: float
    :param target_column: Nombre de la columna objetivo
    :type target_column: str
    :param is_stratified: Si es True, mantiene la proporción de clases en la división
    :type is_stratified: bool
    :returns: Tupla con X_train, X_test, y_train, y_test
    :rtype: tuple
    """
    X = dataset.drop(columns=[target_column])
    y = dataset[[target_column]]
    
    if is_stratified:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    return X_train, X_test, y_train, y_test


def standardize_inputs(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Estandariza las características numéricas.

    :param X_train: Conjunto de entrenamiento
    :type X_train: pd.DataFrame
    :param X_test: Conjunto de prueba
    :type X_test: pd.DataFrame
    :returns: Tupla con X_train_scaled, X_test_scaled
    :rtype: tuple
    """
    # Identificar columnas numéricas
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Inicializar el escalador
    scaler = StandardScaler()
    
    # Estandarizar solo las columnas numéricas
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    return X_train_scaled, X_test_scaled


def elo_expect(Ra, Rb, home_adv):
    return 1.0 / (1.0 + 10.0 ** ((Rb - (Ra + home_adv)) / 400.0))

def compute_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Parámetros 
    R0 = 1500.0
    K_base = 20.0
    HOME_ADV = 80.0           # ventaja local en puntos ELO
    half_life_years = 2.0     # half-life (2 o 3 años suelen andar muy bien)
    half_life_days  = 365.0 * half_life_years
    N_prev = 20               # ← cantidad de partidos “hacia atrás” por equipo

    # Pesos por torneo (ajustables). Default = 1.1.
    weights = {
        "FIFA World Cup": 1.6,
        "UEFA European Championship": 1.4,
        "Copa América": 1.4,
        "FIFA World Cup qualification": 1.3,
        "UEFA Nations League": 1.2
    }
    
    """Calcula ELO de forma cronológica."""
    ratings, last_played, recent_counts = {}, {}, {}
    snapshots = []

    df = df.sort_values("date").copy()
    for _, row in df.iterrows():
        d, ht, at = row["date"], row["home_team"], row["away_team"]
        hs, as_, tour, neut = row["home_score"], row["away_score"], row["tournament"], bool(row["neutral"])

        for team in (ht, at):
            if team not in ratings: ratings[team] = R0
            if team not in recent_counts: recent_counts[team] = deque(maxlen=N_prev)
            if team in last_played:
                delta_days = (d - last_played[team]).days
                if delta_days > 0:
                    decay = 0.5 ** (delta_days / half_life_days)
                    ratings[team] = R0 + (ratings[team] - R0) * decay

        Rh_pre, Ra_pre = ratings[ht], ratings[at]  # ratings antes del partido
        home_adv = 0 if neut else HOME_ADV
        Eh = elo_expect(Rh_pre, Ra_pre, home_adv)
        Ea = 1 - Eh

        # Resultado
        if hs > as_: Sh, Sa = 1, 0
        elif hs < as_: Sh, Sa = 0, 1
        else: Sh, Sa = 0.5, 0.5

        margin = max(1, log(1 + abs(hs - as_), 2))
        Kw_base = K_base * weights.get(tour, 1.1) * margin
        cnt_h, cnt_a = len(recent_counts[ht]), len(recent_counts[at])
        factor_h, factor_a = min(1, cnt_h / N_prev), min(1, cnt_a / N_prev)
        Kw_h, Kw_a = Kw_base * (0.5 + 0.5 * factor_h), Kw_base * (0.5 + 0.5 * factor_a)

        # Actualizamos después del partido
        ratings[ht] = Rh_pre + Kw_h * (Sh - Eh)
        ratings[at] = Ra_pre + Kw_a * (Sa - Ea)

        recent_counts[ht].append(d)
        recent_counts[at].append(d)
        last_played[ht], last_played[at] = d, d

        # Guardamos el snapshot usando los ratings previos al partido
        snapshots.append({
            "date": d,
            "home_team": ht,
            "away_team": at,
            "home_rating": Rh_pre,
            "away_rating": Ra_pre
        })

    elo_df = pd.DataFrame(snapshots)
    elo_df["rating_diff"] = elo_df["home_rating"] - elo_df["away_rating"]
    return elo_df


def setup_mlflow(experiment_name="etl_ml_pipeline"):
    # Conectar con el servidor MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Crear o obtener el experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project":"fifa-2026-win-nowin",
                "team":"mlops1-fiuba" 
                }
            )
    else:
        experiment_id = experiment.experiment_id
        
    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id

def main():
    experiment_id = setup_mlflow()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="fifa_etl_pipeline"):

        print("1. Loading and preprocessing data...")
        data = load_data_from_s3("data", "datasets/results.csv") # file in archive
        data = preprocess_features(data)
        
        # Log parameters
        mlflow.log_param("min_year", MIN_YEAR)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        print("2. Processing features...")
        numerical_features = ["neutral", "year", "month", "dayofweek", "rating_diff"]
        categorical_features = ["home_team", "away_team", "tournament"]
        target_column = "target"
        
        # Log feature information
        mlflow.log_param("numerical_features", numerical_features)
        mlflow.log_param("categorical_features", categorical_features)
        
        # Select features
        selected_data = select_features(
            data,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_column=target_column
        )
        
        # Process categorical variables
        data_with_dummies = make_dummies_variables(selected_data, categorical_features)
        
        # 3. Split data
        print("3 Splitting dataset...")
        X_train, X_test, y_train, y_test = split_dataset(
            data_with_dummies, 
            test_size=0.2, 
            target_column=target_column
        )
        
        # 4. Standardize features
        print("4. Standardizing features...")
        X_train_scaled, X_test_scaled = standardize_inputs(X_train, X_test)
        
        # 5. Save processed data
        print("5. Saving processed data...")
        
        # Save the DataFrames to S3
        save_df_to_s3(X_train, "data", "processed/X_train.csv")
        save_df_to_s3(X_test, "data", "processed/X_test.csv")
        save_df_to_s3(y_train, "data", "processed/y_train.csv")
        save_df_to_s3(y_test, "data", "processed/y_test.csv")
        save_df_to_s3(X_train_scaled, "data", "processed/X_train_scaled.csv")
        save_df_to_s3(X_test_scaled, "data", "processed/X_test_scaled.csv")
        
        # Log metrics
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("class_balance", y_train.mean())
        
        # Log artifacts
        # Log S3 locations to MLflow
        s3_base = "s3://data"
        mlflow.log_param("s3_train_data", f"{s3_base}/processed/X_train.csv")
        mlflow.log_param("s3_test_data", f"{s3_base}/processed/X_test.csv")
        mlflow.log_param("s3_train_labels", f"{s3_base}/processed/y_train.csv")
        mlflow.log_param("s3_test_labels", f"{s3_base}/processed/y_test.csv")
        mlflow.log_param("s3_train_scaled", f"{s3_base}/processed/X_train_scaled.csv")
        mlflow.log_param("s3_test_scaled", f"{s3_base}/processed/X_test_scaled.csv")
        mlflow.log_param("s3_log_columns_dummies", "s3://data/output/log_columns_dummies.txt")
        
        print("ETL pipeline completed successfully!")
        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test
        }
    
if __name__ == "__main__":
    
    # Run the pipeline
    main()
        