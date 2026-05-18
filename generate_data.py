import os
import numpy as np
import pandas as pd
import kagglehub

from config import FEATURES, TARGET, DATA_PATH

SOURCE_TO_CANONICAL = {
    "Age": "age",
    "Weight": "weight",
    "Body_Temperature": "body_temperature",
    "Heart_Rate": "heart_rate",
    "Appetite_Loss": "appetite_loss",
    "Vomiting": "vomiting",
    "Diarrhea": "diarrhea",
    "Coughing": "coughing",
    "Disease_Prediction": "disease",
}


def augment_data(df: pd.DataFrame, n_times: int = 3) -> pd.DataFrame:
    augmented = []
    for _ in range(n_times):
        noisy = df.copy()

        for col in ["age", "weight", "body_temperature", "heart_rate"]:
            noise = np.random.normal(0, 0.05, size=len(df))
            noisy[col] = noisy[col].astype(float) * (1 + noise)

        for col in ["appetite_loss", "vomiting", "diarrhea", "coughing"]:
            flip = np.random.rand(len(df)) < 0.05
            noisy.loc[flip, col] = 1 - noisy.loc[flip, col]

        augmented.append(noisy)

    return pd.concat(augmented, ignore_index=True)


def generate_healthy(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.random.randint(2, 10, n),
            "weight": np.random.normal(500, 50, n),
            "body_temperature": np.random.normal(38.5, 0.3, n),
            "heart_rate": np.random.normal(70, 8, n),
            "appetite_loss": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "vomiting": np.random.choice([0, 1], n, p=[0.98, 0.02]),
            "diarrhea": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "coughing": np.random.choice([0, 1], n, p=[0.94, 0.06]),
            "disease": "healthy",
        }
    )


def build_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")
    csv_path = os.path.join(path, "cleaned_animal_disease_prediction.csv")
    df = pd.read_csv(csv_path)

    df = df[df["Animal_Type"] == "Cow"].copy()
    df = df.rename(columns=SOURCE_TO_CANONICAL)
    df = df.replace({"Yes": 1, "No": 0})

    df["body_temperature"] = (
        df["body_temperature"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    )

    sick_df = df[FEATURES].copy()
    sick_df[TARGET] = "sick"
    sick_df = augment_data(sick_df, n_times=5)
    healthy_df = generate_healthy(len(sick_df))

    final_df = pd.concat([sick_df, healthy_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df.insert(0, "cow_id", [f"COW-{i:04d}" for i in range(1, len(final_df) + 1)])
    return final_df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = build_dataset()
    data.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}")
    print(data.head())
