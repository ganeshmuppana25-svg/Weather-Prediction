"""WeatherPredict - Flask backend for the Decision Tree demo."""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)

FEATURE_ORDER = [
    "temperature", "humidity", "wind_speed", "wind_direction", "cloud_cover",
    "pressure", "visibility", "precipitation", "uv_index", "dew_point"
]
VALID_RANGES = {
    "temperature": (-20, 50), "humidity": (0, 100), "wind_speed": (0, 120),
    "wind_direction": (0, 360), "cloud_cover": (0, 100), "pressure": (950, 1050),
    "visibility": (0, 30), "precipitation": (0, 50), "uv_index": (0, 14),
    "dew_point": (-20, 40),
}
WEATHER_DESCRIPTIONS = {
    "Sunny": "Clear skies with bright sunshine. Great for outdoor activities.",
    "Partly Cloudy": "A mix of sunshine and clouds with generally pleasant conditions.",
    "Cloudy": "Overcast skies with limited sunshine and mild conditions.",
    "Rainy": "Rain is likely. Carry an umbrella and plan outdoor activities carefully.",
    "Thunderstorm": "Stormy conditions with heavy rain and possible lightning. Stay safe indoors.",
    "Foggy": "Reduced visibility due to fog. Travel carefully and use headlights.",
}
WEATHER_ICONS = {
    "Sunny": "☀️", "Partly Cloudy": "⛅", "Cloudy": "☁️",
    "Rainy": "🌧️", "Thunderstorm": "⛈️", "Foggy": "🌫️",
}

model = None
label_encoder = None
metrics = None


def load_artifacts():
    global model, label_encoder, metrics
    model = joblib.load(MODEL_DIR / "weather_model.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    metrics = joblib.load(MODEL_DIR / "metrics.pkl")
    print("Model, encoder and metrics loaded successfully.")


def validate_input(data):
    if not isinstance(data, dict):
        return False, "Please provide weather input data.", None
    values = []
    for feature in FEATURE_ORDER:
        if feature not in data or data[feature] in (None, ""):
            return False, f"Please enter a value for {feature.replace('_', ' ').title()}.", None
        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            return False, f"{feature.replace('_', ' ').title()} must be a number.", None
        if not np.isfinite(value):
            return False, f"{feature.replace('_', ' ').title()} must be a valid number.", None
        low, high = VALID_RANGES[feature]
        if not low <= value <= high:
            return False, f"{feature.replace('_', ' ').title()} must be between {low} and {high}.", None
        values.append(value)
    return True, None, values


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"success": True, "model_loaded": model is not None and label_encoder is not None})


@app.post("/api/predict")
def predict():
    if model is None or label_encoder is None:
        return jsonify({"success": False, "error": "Model is not loaded. Run train_model.py first."}), 500
    try:
        valid, error, values = validate_input(request.get_json(silent=True))
        if not valid:
            return jsonify({"success": False, "error": error}), 400

        row = np.asarray(values, dtype=float).reshape(1, -1)
        encoded_prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        prediction = str(label_encoder.inverse_transform([encoded_prediction])[0])
        confidence = float(probabilities[encoded_prediction] * 100)
        all_probabilities = {
            str(label_encoder.inverse_transform([i])[0]): round(float(probabilities[i] * 100), 2)
            for i in range(len(probabilities))
        }
        summary = {feature: values[i] for i, feature in enumerate(FEATURE_ORDER)}

        return jsonify({
            "success": True,
            "prediction": prediction,
            "icon": WEATHER_ICONS[prediction],
            "confidence": round(confidence, 2),
            "description": WEATHER_DESCRIPTIONS[prediction],
            "all_probabilities": all_probabilities,
            "input_summary": summary,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": f"Prediction error: {exc}"}), 500


@app.get("/api/metrics")
def get_metrics():
    if metrics is None:
        return jsonify({"success": False, "error": "Model metrics are not available."}), 500
    return jsonify({"success": True, "metrics": metrics})


@app.get("/api/dataset")
def get_dataset_info():
    if metrics is None:
        return jsonify({"success": False, "error": "Dataset information is not available."}), 500
    return jsonify({
        "success": True,
        "total_records": metrics["total_records"],
        "num_features": metrics["num_features"],
        "num_classes": metrics["num_classes"],
        "classes": metrics["classes"],
        "features": metrics["feature_names"],
    })


try:
    load_artifacts()
except Exception as exc:
    print(f"WARNING: Could not load model artifacts: {exc}")


if __name__ == "__main__":
    if model is None:
        raise SystemExit("Model files are missing. Run: python train_model.py")
    app.run(host="0.0.0.0", port=5000, debug=True)
