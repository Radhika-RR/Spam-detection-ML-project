
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load trained model & vectorizer
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/spam_phishing_detector.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    vectorized = vectorizer.transform([message])
    prediction = model.predict(vectorized)[0]

    return jsonify({"prediction": "Spam" if prediction == 1 else "Not Spam"})

if __name__ == "__main__":
    app.run(debug=True)
