import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the test data (data the model has NEVER seen)
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Load the trained model we saved in Step 4
model = joblib.load('models/iris_rf_v1.pkl')

# Ask the model to predict species for the test data
y_pred = model.predict(X_test)

# Compare predictions vs actual answers
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))