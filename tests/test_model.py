import joblib
import pandas as pd
import os

def test_model_file_exists():
    """Check that the trained model file was actually saved"""
    assert os.path.exists('models/iris_rf_v1.pkl'), "Model file not found!"

def test_model_loads():
    """Check that the model can be loaded without errors"""
    model = joblib.load('models/iris_rf_v1.pkl')
    assert model is not None

def test_model_predicts_valid_output():
    """Check the model gives a valid prediction for a sample input"""
    model = joblib.load('models/iris_rf_v1.pkl')
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                           columns=['sepal length (cm)', 'sepal width (cm)', 
                                    'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(sample)
    assert prediction[0] in [0, 1, 2], "Prediction is not a valid species (0, 1, or 2)"