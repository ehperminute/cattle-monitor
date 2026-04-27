import os
import joblib
import pandas as pd
from flask import Flask, render_template, abort

from config import FEATURES, MODEL_PATH, MONITORING_DATA_PATH

app = Flask(__name__)

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


def classify_status(sick_probability: float) -> str:
    if sick_probability >= 70:
        return "High risk"
    if sick_probability >= 40:
        return "Review"
    return "Normal"


def build_recommendation(row) -> str:
    temp = row["Body_Temperature"]
    cough = row["Coughing"]
    diarrhea = row["Diarrhea"]
    appetite_loss = row["Appetite_Loss"]
    hr = row["Heart_Rate"]
    sick_probability = row["Sick_Probability"]

    if temp > 39.5 and cough == 1:
        return "Respiratory follow-up recommended."
    if diarrhea == 1 and appetite_loss == 1:
        return "Digestive and hydration review recommended."
    if temp > 39.5 and hr > 85:
        return "General clinical review recommended."
    if sick_probability >= 40:
        return "Veterinary review and closer monitoring recommended."
    return "Continue routine monitoring."


def prepare_dashboard_data():
    if model is None:
        raise RuntimeError("Model file not found. Run train.py first.")
    if not os.path.exists(MONITORING_DATA_PATH):
        raise RuntimeError(
            f"Monitoring dataset not found at {MONITORING_DATA_PATH}. Run generate_monitoring_data.py first."
        )

    df = pd.read_csv(MONITORING_DATA_PATH).copy()
    df["Observation_Date"] = pd.to_datetime(df["Observation_Date"])

    X = df[FEATURES]
    predictions = model.predict(X)

    probs = model.predict_proba(X)
    classes = list(model.classes_)

    healthy_idx = classes.index("healthy")
    sick_idx = classes.index("sick")

    df["Prediction"] = predictions
    df["Healthy_Probability"] = (probs[:, healthy_idx] * 100).round(2)
    df["Sick_Probability"] = (probs[:, sick_idx] * 100).round(2)
    df["Status"] = df["Sick_Probability"].apply(classify_status)
    df["Recommendation"] = df.apply(build_recommendation, axis=1)

    return df


@app.route("/")
def index():
    try:
        df = prepare_dashboard_data()

        latest = (
            df.sort_values(["Cow_ID", "Observation_Date"])
              .groupby("Cow_ID", as_index=False)
              .tail(1)
              .sort_values("Sick_Probability", ascending=False)
        )

        summary = {
            "total_cows": latest["Cow_ID"].nunique(),
            "high_risk": (latest["Status"] == "High risk").sum(),
            "review": (latest["Status"] == "Review").sum(),
            "normal": (latest["Status"] == "Normal").sum(),
        }

        cows = latest.to_dict(orient="records")
        return render_template("index.html", cows=cows, summary=summary, error=None)

    except Exception as exc:
        return render_template("index.html", cows=[], summary=None, error=str(exc))


@app.route("/cow/<cow_id>")
def cow_detail(cow_id):
    try:
        df = prepare_dashboard_data()
        history = df[df["Cow_ID"] == cow_id].sort_values("Observation_Date", ascending=False)

        if history.empty:
            abort(404)

        latest = history.iloc[0].to_dict()
        history_records = history.to_dict(orient="records")

        return render_template(
            "cow_detail.html",
            cow=latest,
            history=history_records
        )
    except Exception as exc:
        abort(500, description=str(exc))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
