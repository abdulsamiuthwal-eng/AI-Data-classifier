"""
DecodeLabs — Project 2: Data Classification Using AI
Prediction Module
"""

import joblib
import numpy as np


def load_model():
    """Load saved model and scaler."""
    model  = joblib.load('model/classifier.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict iris species from measurements.

    Args:
        sepal_length (float): Sepal length in cm
        sepal_width  (float): Sepal width in cm
        petal_length (float): Petal length in cm
        petal_width  (float): Petal width in cm

    Returns:
        dict: prediction result with class, confidence, and probabilities
    """
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    class_colors = {
        'Setosa':     '#FACC15',   # Lemon Yellow
        'Versicolor': '#EAB308',   # Gold / Bright Yellow
        'Virginica':  '#CA8A04'    # Warm Amber Gold
    }
    class_images = {
        'Setosa':     '/static/images/setosa.png',
        'Versicolor': '/static/images/versicolor.png',
        'Virginica':  '/static/images/virginica.png'
    }

    model, scaler = load_model()

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    predicted_class = class_names[prediction]
    confidence = round(float(probabilities[prediction]) * 100, 2)

    prob_dict = {
        class_names[i]: round(float(p) * 100, 2)
        for i, p in enumerate(probabilities)
    }

    return {
        'class':         predicted_class,
        'confidence':    confidence,
        'probabilities': prob_dict,
        'color':         class_colors[predicted_class],
        'image':         class_images[predicted_class]
    }
