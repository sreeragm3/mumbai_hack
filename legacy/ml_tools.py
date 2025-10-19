import joblib
import numpy as np
from typing import Dict, Any
from langchain_core.tools import tool
import os

# Load the trained model and info
model_path = 'models/house_price_model.pkl'
info_path = 'models/model_info.pkl'

if os.path.exists(model_path) and os.path.exists(info_path):
    model = joblib.load(model_path)
    model_info = joblib.load(info_path)
else:
    model = None
    model_info = None

@tool
def predict_house_price(bedrooms: int, bathrooms: int, square_feet: int, age: int) -> str:
    """
    Predicts house price category based on property features.
    
    Args:
        bedrooms: Number of bedrooms (1-5)
        bathrooms: Number of bathrooms (1-3) 
        square_feet: Total square footage (800-3500)
        age: Age of the house in years (0-50)
    
    Returns:
        String describing the predicted price category and confidence
    """
    if model is None:
        return "Error: ML model not found. Please run train_model.py first."
    
    try:
        # Prepare input data
        features = np.array([[bedrooms, bathrooms, square_feet, age]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        confidence = max(prediction_proba)
        
        # Map prediction to category
        category_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        predicted_category = category_map[prediction]
        
        result = f"""
        Based on the property features:
        - Bedrooms: {bedrooms}
        - Bathrooms: {bathrooms}
        - Square Feet: {square_feet}
        - Age: {age} years
        
        Predicted Price Category: {predicted_category}
        Confidence: {confidence:.2%}
        
        The model predicts this house falls into the '{predicted_category}' price category.
        """
        
        return result.strip()
        
    except Exception as e:
        return f"Error making prediction: {str(e)}"

@tool
def get_model_info() -> str:
    """
    Returns information about the machine learning model and its capabilities.
    """
    if model_info is None:
        return "Error: Model information not available."
    
    info = f"""
    Machine Learning Model Information:
    - Model Type: Random Forest Classifier
    - Features: {', '.join(model_info['features'])}
    - Prediction Categories: {', '.join(model_info['target_classes'])}
    - Purpose: Predicts house price categories based on property features
    
    To make a prediction, provide:
    1. Number of bedrooms (1-5)
    2. Number of bathrooms (1-3)
    3. Square footage (800-3500)
    4. Age of house in years (0-50)
    """
    
    return info
