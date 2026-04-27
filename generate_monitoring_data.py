import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import DATA_PATH, MONITORING_DATA_PATH

np.random.seed(42)

NUM_COWS = 10
NUM_DAYS = 12
NUM_SICK_COWS = 2


def clamp(value, low, high):
    return max(low, min(high, value))


def make_date(day_index: int) -> str:
    start_date = datetime.today().date() - timedelta(days=NUM_DAYS - 1)
    return str(start_date + timedelta(days=day_index))


def build_monitoring_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Base dataset not found at {DATA_PATH}. Run generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    healthy_pool = df[df["Disease"] == "healthy"].copy()
    sick_pool = df[df["Disease"] == "sick"].copy()

    if healthy_pool.empty or sick_pool.empty:
        raise ValueError("The source dataset must contain both healthy and sick rows.")

    rows = []
    sick_cow_ids = set(np.random.choice(range(1, NUM_COWS + 1), size=NUM_SICK_COWS, replace=False))

    for cow_num in range(1, NUM_COWS + 1):
        cow_id = f"COW-{cow_num:04d}"

        base_healthy = healthy_pool.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]

        sick_window = None
        sick_template = None

        if cow_num in sick_cow_ids:
            start = np.random.randint(4, 8)
            length = np.random.randint(2, 5)
            end = min(start + length - 1, NUM_DAYS - 1)
            sick_window = (start, end)
            sick_template = sick_pool.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]

        for day_index in range(NUM_DAYS):
            date_str = make_date(day_index)

            age = int(base_healthy["Age"])
            weight = float(base_healthy["Weight"]) + np.random.normal(0, 4)
            temp = float(base_healthy["Body_Temperature"]) + np.random.normal(0, 0.15)
            hr = float(base_healthy["Heart_Rate"]) + np.random.normal(0, 3)

            appetite_loss = int(np.random.choice([0, 1], p=[0.95, 0.05]))
            vomiting = int(np.random.choice([0, 1], p=[0.98, 0.02]))
            diarrhea = int(np.random.choice([0, 1], p=[0.95, 0.05]))
            coughing = int(np.random.choice([0, 1], p=[0.94, 0.06]))
            disease = "healthy"

            if sick_window and sick_window[0] <= day_index <= sick_window[1]:
                # Progression factor inside sick window
                progress = (day_index - sick_window[0] + 1) / (sick_window[1] - sick_window[0] + 1)

                temp = float(base_healthy["Body_Temperature"]) + np.random.uniform(0.8, 1.6) * progress
                hr = float(base_healthy["Heart_Rate"]) + np.random.uniform(8, 18) * progress
                weight = float(base_healthy["Weight"]) + np.random.normal(-3 * progress, 4)

                appetite_loss = int(sick_template["Appetite_Loss"]) if np.random.rand() < 0.8 else appetite_loss
                vomiting = int(sick_template["Vomiting"]) if np.random.rand() < 0.5 else vomiting
                diarrhea = int(sick_template["Diarrhea"]) if np.random.rand() < 0.7 else diarrhea
                coughing = int(sick_template["Coughing"]) if np.random.rand() < 0.7 else coughing
                disease = "sick"

            row = {
                "Cow_ID": cow_id,
                "Observation_Date": date_str,
                "Age": age,
                "Weight": round(clamp(weight, 300, 800), 2),
                "Body_Temperature": round(clamp(temp, 37.5, 41.5), 2),
                "Heart_Rate": round(clamp(hr, 45, 120), 2),
                "Appetite_Loss": appetite_loss,
                "Vomiting": vomiting,
                "Diarrhea": diarrhea,
                "Coughing": coughing,
                "Disease": disease,
            }
            rows.append(row)

    monitoring_df = pd.DataFrame(rows)
    return monitoring_df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    monitoring_df = build_monitoring_dataset()
    monitoring_df.to_csv(MONITORING_DATA_PATH, index=False)

    print(f"Monitoring dataset saved to {MONITORING_DATA_PATH}")
    print(monitoring_df.head(10))
    print("\nRows:", len(monitoring_df))
    print("Unique cows:", monitoring_df['Cow_ID'].nunique())
