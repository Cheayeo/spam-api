from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model once when the server starts
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Spam detector API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    # Error handling
    if data is None:
        return jsonify({"error": "Send JSON like {\"text\": \"your email\"}"}), 400
    if "text" not in data:
        return jsonify({"error": "Missing field: text"}), 400
    if not isinstance(data["text"], str):
        return jsonify({"error": "text must be a string"}), 400
    if data["text"].strip() == "":
        return jsonify({"error": "text cannot be empty"}), 400

    # Predict
    features = tfidf.transform([data["text"]])
    label = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][label])

    return jsonify({
        "prediction": "spam" if label == 1 else "not spam",
        "probability": round(probability, 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
