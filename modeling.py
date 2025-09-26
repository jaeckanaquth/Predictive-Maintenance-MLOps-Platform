import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Constants for paths
DATA_FILE = 'data/transformed_sensor_data.csv'
MODEL_DIR = 'model'
MODEL_PATH = f'{MODEL_DIR}/rf_predictive_maintenance.pkl'

# Load transformed data
df = pd.read_csv(DATA_FILE)
print(f"Loaded data with shape: {df.shape}")

# Verify required columns exist
required_cols = ['Machine failure', 'Product ID', 'UDI']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in data!")

# Prepare features and target
X = df.drop(columns=['Machine failure', 'Product ID', 'UDI'])
y = df['Machine failure']

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"Model trained and saved to {MODEL_PATH}")
