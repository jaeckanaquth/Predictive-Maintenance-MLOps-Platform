import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

DB_FILE = 'storage/sensor_data.db'
MODEL_DIR = 'model'
DATA_DIR = 'data'

def load_raw_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM turbine_sensors", conn)
    conn.close()
    return df

def transform_data(df):
    # Sort by UDI (unique identifier) and Product ID
    df = df.sort_values(by=['UDI', 'Product ID'])
    df.ffill(inplace=True)

    numeric_cols = [
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]',
        'Torque [Nm]', 
        'Tool wear [min]'
    ]
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column {col} not found in input!")

    # Example additional feature: rolling mean using Product ID as group
    df['temp_rolling_mean'] = df.groupby('Product ID')['Air temperature [K]'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    
    # Normalization
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
    print(f"Scaler saved to {MODEL_DIR}/scaler.pkl")
    return df

def save_transformed_data(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(f'{DATA_DIR}/transformed_sensor_data.csv', index=False)
    print(f"Transformed data saved to {DATA_DIR}/transformed_sensor_data.csv")

    # Optional: write back to DB
    conn = sqlite3.connect(DB_FILE)
    df.to_sql('sensor_features', conn, if_exists='replace', index=False)
    conn.close()

raw_df = load_raw_data()
clean_df = transform_data(raw_df)
save_transformed_data(clean_df)
