import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURE_LABELS = [
    "Approved units – 1st sem",
    "Approved units – 2nd sem",
    "Grade – 1st sem",
    "Grade – 2nd sem",
    "Assessments Attempted – 1st sem",
    "Assessments Attempted – 2nd sem",
    "Tuition fees up to date",
    "Scholarship holder",
    "Age at enrollment",
    "Admission grade",
]


def risk_level(score):
    if score < 0.3:
        return "Low"
    elif score < 0.7:
        return "Medium"
    return "High"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_values = [
        float(data["units_1st_approved"]),
        float(data["units_2nd_approved"]),
        float(data["grade_1st"]),
        float(data["grade_2nd"]),
        float(data["evals_1st"]),
        float(data["evals_2nd"]),
        float(data["tuition"]),
        float(data["scholarship"]),
        float(data["age"]),
        float(data["admission_grade"]),
    ]
    input_array = np.array(input_values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    risk_score = float(model.predict_proba(scaled_input)[0][1])

    coefs = model.coef_[0]
    z_scores = scaled_input[0]
    impact = coefs * z_scores
    top_indices = np.argsort(np.abs(impact))[::-1][:3]

    factors = []
    for i in top_indices:
        direction = "increases risk" if impact[i] > 0 else "decreases risk"
        factors.append(f"{FEATURE_LABELS[i]} ({direction})")

    return jsonify({
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level(risk_score),
        "factors": factors,
    })


if __name__ == "__main__":
    app.run(debug=False, port=4000)
