import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

DB_FILE = 'storage/sensor_data.db'

def load_raw_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM turbine_sensors", conn)
    conn.close()
    return df

def transform_data(df):
    # Fill missing values
    df.ffill(inplace=True)
    
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                    'Torque [Nm]', 'Tool wear [min]']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save scaler for later use in API
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Scaler saved to model/scaler.pkl")
    
    return df

def save_transformed_data(df):
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transformed_sensor_data.csv', index=False)
    print("Transformed data saved to data/transformed_sensor_data.csv")

raw_df = load_raw_data()
clean_df = transform_data(raw_df)
save_transformed_data(clean_df)
