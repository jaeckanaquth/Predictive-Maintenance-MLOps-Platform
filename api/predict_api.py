from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained Random Forest model and scaler once on startup
model = joblib.load('model/rf_predictive_maintenance.pkl')
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
    Machine_failure: int  # Usually not needed for prediction inputs; consider removing if unreachable
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
    try:
        # Map input data to feature names expected by the model
        input_df = pd.DataFrame([{
            'UDI': data.UDI,
            'Product ID': data.Product_ID,
            'Type': data.Type,
            'Air temperature [K]': data.Air_temperature_K,
            'Process temperature [K]': data.Process_temperature_K,
            'Rotational speed [rpm]': data.Rotational_speed_rpm,
            'Torque [Nm]': data.Torque_Nm,
            'Tool wear [min]': data.Tool_wear_min,
            'TWF': data.TWF,
            'HDF': data.HDF,
            'PWF': data.PWF,
            'OSF': data.OSF,
            'RNF': data.RNF,
        }])

        numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                        'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

        # Scale numeric features
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Drop columns not used for prediction
        features_to_use = input_df.drop(columns=['UDI', 'Product ID'])

        # Predict machine failure (0 or 1)
        prediction = model.predict(features_to_use)[0]

        return {"machine_failure_prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
