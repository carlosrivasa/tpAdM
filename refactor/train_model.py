import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def load_training_data(X_train_path, y_train_path):
    """
    Carge el train set desde archivos CSV.
    
    Args:
        X_train_path: Path to the training features CSV file
        y_train_path: Path to the training labels CSV file
        
    Returns:
        tuple: (X_train, y_train) as numpy arrays
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)['target'].values
    return X_train, y_train

def train_model(X_train, y_train):
    """
    Entrenamiento del modelo RandomForest model con parámetros fijos.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(
        n_estimators=200,
        criterion='entropy',
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    print("Training RandomForest model...")
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    
    base_path = Path(__file__).parent.parent 
    X_train_path = base_path / 'output' / 'X_train_scaled.csv'
    y_train_path = base_path / 'output' / 'y_train.csv'
    model_path = base_path / 'models' / 'best_rf_model.pkl'
    
    # Create models directory if it doesn't exist
    model_path.parent.mkdir(exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_training_data(X_train_path, y_train_path)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"\nModel successfully trained and saved to {model_path}")