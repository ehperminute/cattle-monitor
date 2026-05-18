import os

import joblib
import pandas as pd
from flask import Flask, abort, render_template, request
from tensorflow.keras.models import load_model

from config import (
    DEFAULT_LANG,
    FEATURES,
    MODEL_PATH,
    MONITORING_DATA_PATH,
    SCALER_PATH,
    SUPPORTED_LANGS,
)
from i18n import get_supported_language_labels, get_translations

app = Flask(__name__)

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


def get_lang() -> str:
    lang = request.args.get("lang", DEFAULT_LANG).strip().lower()
    if lang not in SUPPORTED_LANGS:
        return DEFAULT_LANG
    return lang


def classify_status(sick_probability: float) -> str:
    if sick_probability >= 70:
        return "high_risk"
    if sick_probability >= 40:
        return "review"
    return "normal"


def build_recommendation(row, t):
    symptom_count = row["appetite_loss"] + row["vomiting"] + row["diarrhea"] + row["coughing"]

    if row["body_temperature"] >= 40.0 and row["coughing"] == 1:
        return t.get("respiratory_review", "Respiratory follow-up recommended.")
    if row["diarrhea"] == 1 and row["appetite_loss"] == 1:
        return t.get("digestive_review", "Digestive and hydration review recommended.")
    if row["body_temperature"] >= 39.3 and row["heart_rate"] >= 82:
        return t.get("general_review", "General clinical review recommended.")
    if row["sick_probability"] >= 40:
        return t.get("vet_review", "Veterinary review and closer monitoring recommended.")
    if symptom_count >= 1:
        return t.get("general_review", "General clinical review recommended.")
    return t.get("routine_monitoring", "Continue routine monitoring.")


def build_risk_explanation(row: pd.Series, t: dict) -> str:
    reasons = []
    if row["body_temperature"] >= 39.3:
        reasons.append(t["reason_temp"])
    if row["heart_rate"] >= 82:
        reasons.append(t["reason_hr"])
    if row["coughing"] == 1:
        reasons.append(t["reason_cough"])
    if row["diarrhea"] == 1:
        reasons.append(t["reason_diarrhea"])
    if row["appetite_loss"] == 1:
        reasons.append(t["reason_appetite"])

    if not reasons:
        return t["reason_none"]
    return "; ".join(reasons)


def prepare_dashboard_data(t: dict) -> pd.DataFrame:
    if model is None or scaler is None:
        raise RuntimeError("Model or scaler not found. Run train_sequential.py first.")
    if not os.path.exists(MONITORING_DATA_PATH):
        raise RuntimeError("Monitoring dataset not found. Run generate_monitoring_data.py first.")

    df = pd.read_csv(MONITORING_DATA_PATH).copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    X_scaled = scaler.transform(df[FEATURES])
    sick_probabilities = model.predict(X_scaled, verbose=0).flatten()

    df["prediction"] = ["sick" if prob >= 0.5 else "healthy" for prob in sick_probabilities]
    df["sick_probability"] = (sick_probabilities * 100).round(2)
    df["healthy_probability"] = (100 - df["sick_probability"]).round(2)
    df["status"] = df["sick_probability"].apply(classify_status)
    df["recommendation"] = df.apply(lambda row: build_recommendation(row, t), axis=1)
    df["risk_explanation"] = df.apply(lambda row: build_risk_explanation(row, t), axis=1)
    return df


@app.route("/")
def index():
    lang = get_lang()
    t = get_translations(lang)
    langs = get_supported_language_labels()

    try:
        df = prepare_dashboard_data(t)
        latest = (
            df.sort_values(["cow_id", "observation_date"])
            .groupby("cow_id", as_index=False)
            .tail(1)
            .sort_values("sick_probability", ascending=False)
        )

        summary = {
            "total_cows": int(latest["cow_id"].nunique()),
            "high_risk": int((latest["status"] == "high_risk").sum()),
            "review": int((latest["status"] == "review").sum()),
            "normal": int((latest["status"] == "normal").sum()),
        }

        return render_template(
            "index.html",
            cows=latest.to_dict(orient="records"),
            summary=summary,
            error=None,
            t=t,
            lang=lang,
            langs=langs,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            cows=[],
            summary=None,
            error=str(exc),
            t=t,
            lang=lang,
            langs=langs,
        )


@app.route("/cow/<cow_id>")
def cow_detail(cow_id: str):
    lang = get_lang()
    t = get_translations(lang)
    langs = get_supported_language_labels()

    try:
        df = prepare_dashboard_data(t)
        history = df[df["cow_id"] == cow_id].sort_values("observation_date", ascending=False)
        if history.empty:
            abort(404)

        latest = history.iloc[0].copy()
        previous = history.iloc[1].copy() if len(history) > 1 else None

        if previous is not None:
            trend = {
                "temperature_delta": round(float(latest["body_temperature"] - previous["body_temperature"]), 2),
                "risk_delta": round(float(latest["sick_probability"] - previous["sick_probability"]), 2),
                "previous_temperature": float(previous["body_temperature"]),
                "previous_risk": float(previous["sick_probability"]),
            }
        else:
            trend = {
                "temperature_delta": 0.0,
                "risk_delta": 0.0,
                "previous_temperature": float(latest["body_temperature"]),
                "previous_risk": float(latest["sick_probability"]),
            }

        clinical_summary = {
            "temperature": float(latest["body_temperature"]),
            "heart_rate": float(latest["heart_rate"]),
            "appetite_loss": int(latest["appetite_loss"]),
            "vomiting": int(latest["vomiting"]),
            "diarrhea": int(latest["diarrhea"]),
            "coughing": int(latest["coughing"]),
            "symptom_count": int(
                latest["appetite_loss"]
                + latest["vomiting"]
                + latest["diarrhea"]
                + latest["coughing"]
            ),
        }

        return render_template(
            "cow_detail.html",
            cow=latest.to_dict(),
            history=history.to_dict(orient="records"),
            clinical_summary=clinical_summary,
            trend=trend,
            t=t,
            lang=lang,
            langs=langs,
        )
    except Exception as exc:
        abort(500, description=str(exc))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
