from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained Random Forest model
model = joblib.load('model/rf_predictive_maintenance.pkl')

# Load saved scaler for preprocessing (assumes scaler saved during ETL)
scaler = joblib.load('model/scaler.pkl')

class SensorData(BaseModel):
    UDI: int
    Product_ID: str
    Type: int
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    Machine_failure: int
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict/")
def predict_failure(data: SensorData):
    # Convert input data to DataFrame with model feature names expected during training
    input_df = pd.DataFrame([{
        'UDI': data.UDI,
        'Product ID': data.Product_ID,
        'Type': data.Type,
        'Air temperature [K]': data.Air_temperature_K,
        'Process temperature [K]': data.Process_temperature_K,
        'Rotational speed [rpm]': data.Rotational_speed_rpm,
        'Torque [Nm]': data.Torque_Nm,
        'Tool wear [min]': data.Tool_wear_min,
        'Machine failure': data.Machine_failure,
        'TWF': data.TWF,
        'HDF': data.HDF,
        'PWF': data.PWF,
        'OSF': data.OSF,
        'RNF': data.RNF,
    }])

    # Preprocess input with scaler (only numeric columns relevant for model)
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                    'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict failure class
    prediction = model.predict(input_df.drop(columns=['UDI', 'Product ID', 'Machine failure']))[0]

    return {"machine_failure_prediction": int(prediction)}
