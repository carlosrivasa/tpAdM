import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import category_encoders as ce
    
from collections import deque
from math import log

from pathlib import Path

def load_data_from_source(path: str, filename: str) -> pd.DataFrame:
    """
    Carga los datos crudos del archivo de resultados de partidos.

    :param path: Ruta donde está ubicado el archivo CSV con los datos
    :type path: str
    :param filename: Nombre del archivo CSV
    :type filename: str
    :returns: DataFrame con los datos de los partidos
    :rtype: pd.DataFrame
    """
    # Cargar los datos
    return pd.read_csv(path + filename)

def preprocess_features(results: pd.DataFrame, min_year: int = 1920) -> pd.DataFrame:
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
    Path("./output").mkdir(exist_ok=True)
    with open("./output/log_columns_dummies.txt", "w") as f:
        f.write("New columns after target encoding:\n")
        f.write("\n".join(encoded_data.columns.tolist()))
    
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
    
    # Guardar los conjuntos de datos
    X_train.to_csv("./output/X_train.csv", index=False)
    X_test.to_csv("./output/X_test.csv", index=False)
    y_train.to_csv("./output/y_train.csv", index=False)
    y_test.to_csv("./output/y_test.csv", index=False)
    
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
    
    # Guardar los conjuntos escalados
    X_train_scaled.to_csv("./output/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv("./output/X_test_scaled.csv", index=False)
    
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
    
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data_from_source("./archive/", "results.csv")
    data = preprocess_features(data)
    
    numerical_features = ["neutral", "year", "month", "dayofweek", "rating_diff"]
    categorical_features = ["home_team", "away_team", "tournament"]
    target_column = "target"

    # Convert categorical variables

    # 3. Select only the features we need
    data = select_features(
        data,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target_column
    )

    data = make_dummies_variables(data, categorical_features)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = split_dataset(
        data, test_size=0.2, target_column="target"
    )
    
    # Standardize features
    X_train_scaled, X_test_scaled = standardize_inputs(X_train, X_test)
    
    print("X_train_scaled:", X_test_scaled.head())
        