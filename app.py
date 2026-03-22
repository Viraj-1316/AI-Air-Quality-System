from flask import Flask, jsonify
from model_script_new import run_prediction_once_now
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Air Quality API Running 🚀"

@app.route("/predict")
def predict():
    try:
        result = run_prediction_once_now()
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)