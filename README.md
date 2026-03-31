# Student Dropout Risk Prediction

## Problem
Predict whether a student is at risk of dropping out using academic and demographic features.

## Features
- Predicts dropout risk (probability)
- Classifies risk into Low / Medium / High
- Highlights top contributing factors
- REST API using Flask

## Tech Stack
- Python
- Flask
- Scikit-learn (Logistic Regression)
- Pandas

## Model Details
- Features are scaled using StandardScaler
- Model outputs probability using predict_proba
- Feature impact computed using model coefficients
- Logistic Regression used for classification

## How to Run

pip install -r requirements.txt
python app.py

## API Endpoint
POST /predict

## Example Output

{
  "risk_score": 0.62,
  "risk_level": "Medium",
  "factors": [
    "Grade – 1st sem (decreases risk)",
    "Assessments Attempted – 2nd sem (increases risk)"
  ]
}

