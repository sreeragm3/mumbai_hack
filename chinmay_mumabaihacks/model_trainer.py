"""
Machine Learning model trainer for the Hospital Surge Readiness (HSR) Platform.
Handles data loading, feature preparation, model training, and prediction functionality.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Tuple, Dict, Any


def load_data(file_path: str = 'data/historical_records.csv') -> pd.DataFrame:
    """
    Load historical surge data from CSV file.
    
    Args:
        file_path (str): Path to the historical records CSV file
        
    Returns:
        pd.DataFrame: Loaded historical data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the data file is empty or malformed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Historical data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Historical data file is empty")
        
        # Convert date column to datetime
        df['festival_date'] = pd.to_datetime(df['festival_date'])
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for machine learning model.
    
    Args:
        df (pd.DataFrame): Historical surge data
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
    """
    # Target variable: blood_units_used (critical inventory item)
    y = df['blood_units_used'].copy()
    
    # Feature engineering
    X = pd.DataFrame()
    
    # Numerical features
    X['ed_volume_change_percent'] = df['ed_volume_change_percent']
    X['trauma_cases'] = df['trauma_cases']
    
    # One-hot encode festival_name
    festival_dummies = pd.get_dummies(df['festival_name'], prefix='festival')
    X = pd.concat([X, festival_dummies], axis=1)
    
    # Add burn_cases and respiratory_cases as additional features
    X['burn_cases'] = df['burn_cases']
    X['respiratory_cases'] = df['respiratory_cases']
    
    return X, y


def train_and_save_model(X: pd.DataFrame, y: pd.Series, 
                        model_path: str = 'data/surge_predictor.joblib') -> LinearRegression:
    """
    Train a LinearRegression model and save it to disk.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model_path (str): Path to save the trained model
        
    Returns:
        LinearRegression: Trained model
    """
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained - MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # Save the model
    joblib.dump(model, model_path)
    
    return model


def predict_surge_needs(model: LinearRegression, X_cols: list, 
                       festival_name: str, ed_surge_percent: float, 
                       trauma_cases_forecast: int, burn_cases_forecast: int = None,
                       respiratory_cases_forecast: int = None) -> float:
    """
    Predict blood units needed for a specific festival surge scenario.
    
    Args:
        model (LinearRegression): Trained model
        X_cols (list): List of feature column names from training
        festival_name (str): Name of the festival ('Diwali' or 'Holi')
        ed_surge_percent (float): Expected ED volume change percentage
        trauma_cases_forecast (int): Expected number of trauma cases
        burn_cases_forecast (int, optional): Expected number of burn cases
        respiratory_cases_forecast (int, optional): Expected number of respiratory cases
        
    Returns:
        float: Predicted blood units needed
    """
    # Create feature vector
    features = pd.DataFrame(0, index=[0], columns=X_cols)
    
    # Set numerical features
    features['ed_volume_change_percent'] = ed_surge_percent
    features['trauma_cases'] = trauma_cases_forecast
    
    # Set burn and respiratory cases if provided, otherwise use typical values
    if burn_cases_forecast is not None:
        features['burn_cases'] = burn_cases_forecast
    else:
        # Use typical values based on festival
        features['burn_cases'] = 150 if festival_name.lower() == 'diwali' else 10
    
    if respiratory_cases_forecast is not None:
        features['respiratory_cases'] = respiratory_cases_forecast
    else:
        # Use typical values based on festival
        features['respiratory_cases'] = 100 if festival_name.lower() == 'diwali' else 85
    
    # Set festival one-hot encoding
    festival_col = f'festival_{festival_name}'
    if festival_col in features.columns:
        features[festival_col] = 1
    
    # Make prediction and return raw value
    predicted_blood_units = model.predict(features)[0]
    return predicted_blood_units


def get_blood_units_recommendation(blood_units: float) -> str:
    """
    Generate a recommendation based on predicted blood units needed.
    
    Args:
        blood_units (float): Predicted number of blood units
        
    Returns:
        str: Recommendation message
    """
    if blood_units < 25:
        return "Normal blood inventory levels sufficient"
    elif blood_units < 35:
        return "Consider increasing blood inventory by 20-30%"
    elif blood_units < 45:
        return "Significant blood inventory increase needed (40-50%)"
    else:
        return "Critical: Major blood inventory surge required (50%+)"


def load_trained_model(model_path: str = 'data/surge_predictor.joblib') -> LinearRegression:
    """
    Load a previously trained model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        LinearRegression: Loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    return model


if __name__ == "__main__":
    # Main execution block for model training
    print("=" * 60)
    print("HSR Platform - Model Training")
    print("=" * 60)
    
    try:
        # Load data
        df = load_data()
        
        # Prepare features
        X, y = prepare_data(df)
        
        # Train and save model
        model = train_and_save_model(X, y)
        
        print("\n" + "=" * 40)
        print("Model Training Complete!")
        print("=" * 40)
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
