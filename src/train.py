import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the preprocessed training data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Create the model (a "Random Forest" - explained below)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train it - this is where the "learning" happens
model.fit(X_train, y_train.values.ravel())

# Save the trained model to the models/ folder
joblib.dump(model, 'models/iris_rf_v1.pkl')

print("Model trained and saved successfully!")
print(f"Model type: {type(model).__name__}")