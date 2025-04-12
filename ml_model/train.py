import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

def load_weather_data():
    """Load and preprocess weather data from CSV"""
    df = pd.read_csv('ml_model/data/weather_data.csv')
    # Convert timestamp to datetime and extract features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    return df

def prepare_features(df):
    """Prepare enhanced features and target variable"""
    # Create target (next hour's temperature)
    df['target_temp'] = df.groupby('city')['temperature'].shift(-1)
    df = df.dropna()
    
    # Feature engineering
    features = [
        'temperature', 'humidity', 'wind_speed', 'pressure',
        'precipitation', 'cloud_coverage', 'hour', 'day_of_week'
    ]
    
    # One-hot encode city and weather condition
    encoder = OneHotEncoder(sparse_output=False)
    city_encoded = encoder.fit_transform(df[['city']])
    condition_encoded = encoder.fit_transform(df[['weather_condition']])
    
    # Combine all features
    X = np.hstack([
        df[features].values,
        city_encoded,
        condition_encoded
    ])
    
    return train_test_split(X, df['target_temp'], test_size=0.2)

def train_and_save_model():
    """Train model with enhanced features"""
    X_train, X_test, y_train, y_test = prepare_features(load_weather_data())
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Add basic evaluation
    from sklearn.metrics import mean_absolute_error
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained - MAE: {mae:.2f}°C")
    
    joblib.dump(model, 'ml_model/weather_model.pkl')

if __name__ == "__main__":
    train_and_save_model()
