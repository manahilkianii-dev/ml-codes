import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# --- DATA LOADING AND MODEL TRAINING ---

# 1. Load the data
try:
    df = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("Error: heart.csv not found. Make sure the file is in the same directory.")
    exit()

# 2. Prepare the data
X = df.drop('target', axis=1)
y = df['target']

# 3. Train the Classification Model (Random Forest)
print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
# We train on the full dataset here to maximize model performance for the application
model.fit(X, y)
print("Model training complete.")

# 4. Save the trained model
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
    
print(f"Model successfully saved to {model_filename}")