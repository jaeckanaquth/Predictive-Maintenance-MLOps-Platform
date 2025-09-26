import pandas as pd
import sqlite3
import os

CSV_FILE = 'data/ai4i2020.csv'  # Path to your CSV file
DB_FILE = 'storage/sensor_data.db'

os.makedirs('storage', exist_ok=True)

# Load CSV with pandas
df = pd.read_csv(CSV_FILE)

# Map categorical 'Type' to numeric for ease of use later (optional)
df['Type'] = df['Type'].astype('category').cat.codes

# Connect to SQLite and write table (replace table if exists)
conn = sqlite3.connect(DB_FILE)
df.to_sql('turbine_sensors', conn, if_exists='replace', index=False)

print(f"Loaded {len(df)} rows into SQLite database at {DB_FILE}")

conn.close()
