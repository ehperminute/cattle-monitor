import os
import joblib
import pandas as pd

from config import FEATURES, MODEL_PATH, MONITORING_DATA_PATH
from db import init_db, execute, query_one


def classify_status(sick_probability: float) -> str:
    if sick_probability >= 70:
        return "High risk"
    if sick_probability >= 40:
        return "Review"
    return "Normal"


def build_recommendation(row) -> str:
    temp = row["Body_Temperature"]
    hr = row["Heart_Rate"]
    cough = row["Coughing"]
    diarrhea = row["Diarrhea"]
    vomiting = row["Vomiting"]
    appetite_loss = row["Appetite_Loss"]
    sick_probability = row["Sick_Probability"]

    symptom_count = appetite_loss + vomiting + diarrhea + cough

    if temp >= 40.0 and cough == 1:
        return "Urgent respiratory review recommended."
    if diarrhea == 1 and appetite_loss == 1:
        return "Digestive and hydration review recommended."
    if cough == 1 and sick_probability >= 35:
        return "Respiratory follow-up and observation recommended."
    if vomiting == 1 or diarrhea == 1:
        return "Digestive follow-up recommended."
    if temp >= 39.3 and hr >= 82:
        return "General clinical review recommended."
    if sick_probability >= 70:
        return "High-risk case: veterinary review recommended."
    if sick_probability >= 40:
        return "Moderate-risk case: closer monitoring recommended."
    if symptom_count >= 1:
        return "Minor warning signs detected: repeat observation recommended."
    return "Continue routine monitoring."


def main() -> None:
    if not os.path.exists(MONITORING_DATA_PATH):
        raise FileNotFoundError(
            f"Monitoring dataset not found at {MONITORING_DATA_PATH}. Run generate_monitoring_data.py first."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train.py first."
        )

    init_db()

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(MONITORING_DATA_PATH).copy()

    probs = model.predict_proba(df[FEATURES])
    classes = list(model.classes_)
    healthy_idx = classes.index("healthy")
    sick_idx = classes.index("sick")

    df["Prediction"] = model.predict(df[FEATURES])
    df["Healthy_Probability"] = (probs[:, healthy_idx] * 100).round(2)
    df["Sick_Probability"] = (probs[:, sick_idx] * 100).round(2)
    df["Status"] = df["Sick_Probability"].apply(classify_status)
    df["Recommendation"] = df.apply(build_recommendation, axis=1)

    inserted = 0

    for _, row in df.iterrows():
        cow_id = row["Cow_ID"]
        existing_cow = query_one("SELECT cow_id FROM cows WHERE cow_id = ?", (cow_id,))
        if existing_cow is None:
            execute("INSERT INTO cows (cow_id) VALUES (?)", (cow_id,))

        duplicate = query_one(
            """
            SELECT id
            FROM observations
            WHERE cow_id = ? AND observation_date = ?
            """,
            (cow_id, row["Observation_Date"]),
        )
        if duplicate is not None:
            continue

        execute(
            """
            INSERT INTO observations (
                cow_id, observation_date, age, weight, body_temperature, heart_rate,
                appetite_loss, vomiting, diarrhea, coughing, prediction,
                healthy_probability, sick_probability, status, recommendation, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cow_id,
                row["Observation_Date"],
                float(row["Age"]),
                float(row["Weight"]),
                float(row["Body_Temperature"]),
                float(row["Heart_Rate"]),
                int(row["Appetite_Loss"]),
                int(row["Vomiting"]),
                int(row["Diarrhea"]),
                int(row["Coughing"]),
                str(row["Prediction"]),
                float(row["Healthy_Probability"]),
                float(row["Sick_Probability"]),
                str(row["Status"]),
                str(row["Recommendation"]),
                "seed",
            ),
        )
        inserted += 1

    print(f"Seed complete. Inserted {inserted} observations into the SQLite database.")


if __name__ == "__main__":
    main()
