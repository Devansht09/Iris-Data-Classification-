from flask import Flask, request, jsonify, render_template
import numpy as np
from iris_models import svm_model, logreg_model, tree_model, class_names

app = Flask(__name__)

models = {
    "svm": svm_model,
    "logreg": logreg_model,
    "tree": tree_model
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    model = models[data["model"]]

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return jsonify({
        "prediction": prediction,
        "confidence": round(float(max(probabilities)) * 100, 2),
        "probs": dict(zip(class_names, probabilities.round(3).tolist()))
    })

if __name__ == "__main__":
    app.run(debug=True)
