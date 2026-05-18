import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import DATA_PATH, MONITORING_DATA_PATH

np.random.seed(42)

NUM_COWS = 12
NUM_DAYS = 14
NUM_SICK_COWS = 5


def clamp(value, low, high):
    return max(low, min(high, value))


def observation_date(day_index: int) -> str:
    start_date = datetime.today().date() - timedelta(days=NUM_DAYS - 1)
    return str(start_date + timedelta(days=day_index))


def build_monitoring_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Base dataset not found at {DATA_PATH}. Run generate_data.py first.")

    df = pd.read_csv(DATA_PATH)
    healthy_pool = df[df["disease"] == "healthy"].copy()
    sick_pool = df[df["disease"] == "sick"].copy()

    if healthy_pool.empty or sick_pool.empty:
        raise ValueError("The source dataset must contain both healthy and sick rows.")

    rows = []
    sick_cow_ids = set(np.random.choice(range(1, NUM_COWS + 1), size=NUM_SICK_COWS, replace=False))

    for cow_num in range(1, NUM_COWS + 1):
        cow_id = f"COW-{cow_num:04d}"
        base = healthy_pool.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]

        sick_window = None
        sick_template = None
        if cow_num in sick_cow_ids:
            pattern = np.random.choice(["late", "recovering", "mid"])
            if pattern == "late":
                start = NUM_DAYS - np.random.randint(4, 6)
                end = NUM_DAYS - 1
            elif pattern == "recovering":
                start = NUM_DAYS - np.random.randint(7, 9)
                end = NUM_DAYS - 2
            else:
                start = np.random.randint(4, 8)
                end = min(start + np.random.randint(2, 5), NUM_DAYS - 1)
            sick_window = (start, end)
            sick_template = sick_pool.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]

        for day_index in range(NUM_DAYS):
            age = int(base["age"])
            weight = float(base["weight"]) + np.random.normal(0, 4)
            body_temperature = float(base["body_temperature"]) + np.random.normal(0, 0.15)
            heart_rate = float(base["heart_rate"]) + np.random.normal(0, 3)
            appetite_loss = int(np.random.choice([0, 1], p=[0.95, 0.05]))
            vomiting = int(np.random.choice([0, 1], p=[0.98, 0.02]))
            diarrhea = int(np.random.choice([0, 1], p=[0.95, 0.05]))
            coughing = int(np.random.choice([0, 1], p=[0.94, 0.06]))
            disease = "healthy"

            if sick_window and sick_window[0] <= day_index <= sick_window[1]:
                progress = (day_index - sick_window[0] + 1) / (sick_window[1] - sick_window[0] + 1)
                body_temperature = float(base["body_temperature"]) + np.random.uniform(0.8, 1.6) * progress
                heart_rate = float(base["heart_rate"]) + np.random.uniform(8, 18) * progress
                weight = float(base["weight"]) + np.random.normal(-3 * progress, 4)
                appetite_loss = int(sick_template["appetite_loss"]) if np.random.rand() < 0.8 else appetite_loss
                vomiting = int(sick_template["vomiting"]) if np.random.rand() < 0.5 else vomiting
                diarrhea = int(sick_template["diarrhea"]) if np.random.rand() < 0.7 else diarrhea
                coughing = int(sick_template["coughing"]) if np.random.rand() < 0.7 else coughing
                disease = "sick"

            rows.append(
                {
                    "cow_id": cow_id,
                    "observation_date": observation_date(day_index),
                    "age": age,
                    "weight": round(clamp(weight, 300, 800), 2),
                    "body_temperature": round(clamp(body_temperature, 37.5, 41.5), 2),
                    "heart_rate": round(clamp(heart_rate, 45, 120), 2),
                    "appetite_loss": appetite_loss,
                    "vomiting": vomiting,
                    "diarrhea": diarrhea,
                    "coughing": coughing,
                    "disease": disease,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    monitoring_df = build_monitoring_dataset()
    monitoring_df.to_csv(MONITORING_DATA_PATH, index=False)
    print(f"Monitoring dataset saved to {MONITORING_DATA_PATH}")
    print(monitoring_df.head())
