from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Iris Classifier API")

# Load the trained model once, when the app starts
model = joblib.load('models/iris_rf_v1.pkl')

# Map numbers back to real flower names
species_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.get("/")
def home():
    return {"message": "Iris Classifier API is running!"}

@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                           columns=['sepal length (cm)', 'sepal width (cm)',
                                    'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(sample)[0]
    return {
        "prediction": int(prediction),
        "species": species_names[int(prediction)]
    }