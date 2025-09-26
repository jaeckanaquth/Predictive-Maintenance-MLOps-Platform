import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load transformed data
df = pd.read_csv('data/transformed_sensor_data.csv')

# Features and target
X = df.drop(columns=['Machine failure', 'Product ID', 'UDI'])
y = df['Machine failure']

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/rf_predictive_maintenance.pkl')
print("Model trained and saved to model/rf_predictive_maintenance.pkl")
