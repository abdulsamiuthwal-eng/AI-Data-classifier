"""
DecodeLabs — Project 2: Data Classification Using AI
Model Training Pipeline
Author: DecodeLabs Intern
"""

import os
import sys
import json
import joblib

# Fix Windows encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset():
    """Load Iris dataset and return as DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    })
    return df, iris


# ─────────────────────────────────────────────
# 2. PREPROCESS DATA
# ─────────────────────────────────────────────
def preprocess(df):
    """Split and scale data."""
    X = df[['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']].values
    y = df['target'].values

    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# ─────────────────────────────────────────────
# 3. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test):
    """Train KNN, Decision Tree, and Random Forest. Return results."""
    class_names = ['Setosa', 'Versicolor', 'Virginica']

    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree':       DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ''

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
        rec  = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
        f1   = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
        cm   = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy':  acc,
            'precision': prec,
            'recall':    rec,
            'f1_score':  f1,
            'confusion_matrix': cm
        }

        print(f"  [OK] {name}: Accuracy = {acc}%")

        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    return results, best_model, best_model_name


# ─────────────────────────────────────────────
# 4. SAVE MODEL & RESULTS
# ─────────────────────────────────────────────
def save_artifacts(model, scaler, results, best_model_name):
    """Save model, scaler, and results to disk."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)

    joblib.dump(model, os.path.join(base_dir, 'classifier.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))

    output = {
        'best_model': best_model_name,
        'results': results,
        'class_names': ['Setosa', 'Versicolor', 'Virginica'],
        'feature_names': [
            'sepal_length', 'sepal_width',
            'petal_length', 'petal_width'
        ]
    }

    with open(os.path.join(base_dir, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Best Model: {best_model_name}")
    print("[OK] Model saved to classifier.pkl")
    print("[OK] Results saved to results.json")


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_training_pipeline():
    print("="*50)
    print("  DecodeLabs - AI Classification Pipeline")
    print("="*50)

    print("\n[1/4] Loading dataset...")
    df, iris = load_dataset()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Classes: {list(df['species'].unique())}")

    print("\n[2/4] Splitting & scaling data...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("\n[3/4] Training models...")
    results, best_model, best_model_name = train_all_models(
        X_train, X_test, y_train, y_test
    )

    print("\n[4/4] Saving artifacts...")
    save_artifacts(best_model, scaler, results, best_model_name)

    print("\nTraining complete!")
    return results


if __name__ == '__main__':
    run_training_pipeline()
