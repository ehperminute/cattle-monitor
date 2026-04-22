
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

from config import FEATURES, MODEL_PATH

app = Flask(__name__)

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


def parse_checkbox(name: str) -> int:
    return 1 if request.form.get(name) == "on" else 0


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "Model file not found. Train the model first."
            return render_template("index.html", result=result, probability=probability, error=error)

        try:
            row = {
                "Age": float(request.form["Age"]),
                "Weight": float(request.form["Weight"]),
                "Body_Temperature": float(request.form["Body_Temperature"]),
                "Heart_Rate": float(request.form["Heart_Rate"]),
                "Appetite_Loss": parse_checkbox("Appetite_Loss"),
                "Vomiting": parse_checkbox("Vomiting"),
                "Diarrhea": parse_checkbox("Diarrhea"),
                "Coughing": parse_checkbox("Coughing"),
            }

            df_input = pd.DataFrame([row], columns=FEATURES)

            pred = model.predict(df_input)[0]
            result = pred

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_input)[0]
                classes = list(model.classes_)
                probability = {
                    classes[i]: round(float(proba[i]) * 100, 2)
                    for i in range(len(classes))
                }

        except Exception as exc:
            error = f"Invalid input: {exc}"

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error,
    )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
