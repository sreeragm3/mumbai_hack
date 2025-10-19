import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def create_sample_data():
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data for house price prediction
    data = {
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'square_feet': np.random.randint(800, 3500, n_samples),
        'age': np.random.randint(0, 50, n_samples),
    }
    
    # Create target variable (price category: 0=Low, 1=Medium, 2=High)
    df = pd.DataFrame(data)
    df['price_category'] = (
        (df['bedrooms'] * 0.3 + 
         df['bathrooms'] * 0.2 + 
         df['square_feet'] * 0.001 + 
         (50 - df['age']) * 0.01) > 2.5
    ).astype(int)
    
    # Add some randomness
    df.loc[df['price_category'] == 1, 'price_category'] = np.random.choice(
        [1, 2], size=(df['price_category'] == 1).sum(), p=[0.7, 0.3]
    )
    
    return df

def train_model():
    """Train and save the ML model"""
    print("Creating sample data...")
    df = create_sample_data()
    
    # Prepare features and target
    X = df[['bedrooms', 'bathrooms', 'square_feet', 'age']]
    y = df['price_category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/house_price_model.pkl')
    
    # Save feature names for reference
    feature_info = {
        'features': ['bedrooms', 'bathrooms', 'square_feet', 'age'],
        'target_classes': ['Low', 'Medium', 'High']
    }
    joblib.dump(feature_info, 'models/model_info.pkl')
    
    print("Model saved successfully!")
    return model

if __name__ == "__main__":
    train_model()
