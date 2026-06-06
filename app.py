from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "life_expectancy_model.joblib"

app = Flask(__name__)
artifact = joblib.load(MODEL_PATH)

DISPLAY_FIELDS = [
    ("Year", "Tahun", 2015, 2000, 2030, 1),
    ("Adult Mortality", "Adult Mortality", 180, 0, 600, 1),
    ("infant deaths", "Infant Deaths", 10, 0, 200, 1),
    ("Alcohol", "Alcohol", 4.0, 0, 20, 0.1),
    ("percentage expenditure", "Percentage Expenditure", 500.0, 0, 20000, 10),
    ("Hepatitis B", "Hepatitis B", 85, 0, 100, 1),
    ("Measles ", "Measles", 120, 0, 5000, 1),
    (" BMI ", "BMI", 45.0, 5, 90, 0.1),
    ("under-five deaths ", "Under-five Deaths", 15, 0, 250, 1),
    ("Polio", "Polio", 85, 0, 100, 1),
    ("Total expenditure", "Total Expenditure", 6.0, 0, 20, 0.1),
    ("Diphtheria ", "Diphtheria", 85, 0, 100, 1),
    (" HIV/AIDS", "HIV/AIDS", 0.5, 0, 50, 0.1),
    ("GDP", "GDP", 3500.0, 0, 100000, 100),
    ("Population", "Population", 12000000, 0, 1500000000, 1000),
    (" thinness  1-19 years", "Thinness 1-19 Years", 5.0, 0, 40, 0.1),
    (" thinness 5-9 years", "Thinness 5-9 Years", 5.0, 0, 40, 0.1),
    ("Income composition of resources", "Income Composition", 0.65, 0, 1, 0.01),
    ("Schooling", "Schooling", 12.0, 0, 25, 0.1),
]


def transform_payload(payload: dict) -> pd.DataFrame:
    row = {}
    for column in artifact["feature_columns"]:
        if column in artifact["encoders"]:
            encoder = artifact["encoders"][column]
            value = payload.get(column, encoder.classes_[0])
            if value not in encoder.classes_:
                value = encoder.classes_[0]
            row[column] = int(encoder.transform([value])[0])
        else:
            default_value = artifact["fill_values"].get(column, artifact["means"].get(column, 0))
            row[column] = float(payload.get(column, default_value) or default_value)

    frame = pd.DataFrame([row], columns=artifact["feature_columns"])
    frame[artifact["columns_to_scale"]] = artifact["scaler"].transform(
        frame[artifact["columns_to_scale"]]
    )
    return frame


def build_recommendations(payload: dict) -> list[dict[str, str]]:
    rules = [
        ("Schooling", "Tingkatkan rata-rata lama sekolah", "tahun"),
        ("Income composition of resources", "Perkuat komposisi pendapatan dan akses sumber daya", ""),
        ("Diphtheria ", "Naikkan cakupan imunisasi difteri", "%"),
        ("Polio", "Naikkan cakupan imunisasi polio", "%"),
        (" BMI ", "Jaga indikator BMI populasi tetap sehat", ""),
    ]
    recommendations = []
    for column, title, unit in rules:
        current = float(payload.get(column, 0) or 0)
        target = float(artifact["medians"].get(column, current))
        if current < target:
            recommendations.append(
                {
                    "title": title,
                    "detail": f"Nilai sekarang {current:.2f}{unit}; median data latih {target:.2f}{unit}.",
                }
            )
    return recommendations[:3]


@app.route("/")
def index():
    countries = artifact["encoders"]["Country"].classes_.tolist()
    statuses = artifact["encoders"]["Status"].classes_.tolist()
    top_features = artifact["feature_importance"][:5]
    return render_template(
        "index.html",
        countries=countries,
        statuses=statuses,
        fields=DISPLAY_FIELDS,
        top_features=top_features,
        rows=artifact["dataset_rows"],
        columns=artifact["dataset_columns"],
    )


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    frame = transform_payload(payload)
    prediction = float(artifact["model"].predict(frame)[0])
    prediction = max(35.0, min(95.0, prediction))
    return jsonify(
        {
            "life_expectancy": round(prediction, 2),
            "category": "Tinggi" if prediction >= 72 else "Menengah" if prediction >= 60 else "Rendah",
            "recommendations": build_recommendations(payload),
            "top_features": artifact["feature_importance"][:5],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=True)
