from flask import Flask, jsonify
from model_script_new import run_prediction_once_now

app = Flask(__name__)

@app.route("/")
def home():
    return "Air Quality API Running 🚀"

@app.route("/predict")
def predict():
    try:
        result = run_prediction_once_now()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})