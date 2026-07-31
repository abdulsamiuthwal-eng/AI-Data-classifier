# 🧠 AI Data Classification — DecodeLabs Project 2

> **Batch 2026 | Powered by DecodeLabs**  
> A professional AI-powered data classification system using supervised learning.

---

## 🚀 Overview

This project implements a complete **Machine Learning classification pipeline** that:
- Loads the **Iris dataset** (150 samples, 4 features, 3 classes)
- Splits data into **80% training / 20% testing**
- Trains and compares **3 classification algorithms**
- Displays results on a **premium web dashboard**

---

## 🛠️ Tech Stack

| Layer     | Technology           |
|-----------|----------------------|
| Backend   | Python + Flask       |
| ML        | scikit-learn         |
| Data      | pandas + numpy       |
| Frontend  | HTML + CSS + Chart.js|

---

## 🤖 Algorithms Used

| Algorithm       | Description                              |
|-----------------|------------------------------------------|
| KNN (K=5)       | Distance-based classifier                |
| Decision Tree   | Rule-based tree (max depth 4)            |
| Random Forest   | Ensemble of 100 decision trees           |

---

## 📁 Project Structure

```
Project 2/
├── app.py              # Flask web application
├── requirements.txt    # Python dependencies
├── model/
│   ├── train.py        # Training pipeline
│   ├── predict.py      # Prediction module
│   ├── classifier.pkl  # Saved best model
│   ├── scaler.pkl      # Feature scaler
│   └── results.json    # Training results
├── templates/
│   ├── index.html      # Dashboard
│   ├── train.html      # Training page
│   └── predict.html    # Prediction page
└── static/
    └── css/style.css   # Premium dark theme
```

---

## ⚙️ Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Open Browser
```
http://127.0.0.1:5000
```

---

## 📊 Features

- ✅ **Dashboard** — Algorithm comparison with accuracy charts
- ✅ **Train Page** — One-click model training with live results
- ✅ **Predict Page** — Interactive prediction form with confidence scores
- ✅ **Confusion Matrix** — Visual performance breakdown
- ✅ **REST API** — JSON endpoints for predictions
- ✅ **Premium UI** — Dark theme with animations

---

## 🌸 Dataset — Iris

| Feature       | Range       |
|---------------|-------------|
| Sepal Length  | 4.3 – 7.9 cm|
| Sepal Width   | 2.0 – 4.4 cm|
| Petal Length  | 1.0 – 6.9 cm|
| Petal Width   | 0.1 – 2.5 cm|

**Classes:** Setosa 🌸 | Versicolor 🌺 | Virginica 🌻

---

## 🎯 Key Skills Demonstrated

- **Data handling** with pandas
- **Supervised Learning** pipeline
- **Model training & evaluation**
- **Web deployment** with Flask
- **Data visualization** with Chart.js

---

> Built with ❤️ as part of **DecodeLabs AI Internship — Batch 2026**
