"""
import pandas as pd
import numpy as np
import kagglehub
from config import FEATURES

def augment_data(df, n_times=3):
    augmented = []

    for _ in range(n_times):
        noisy = df.copy()

        for col in ["Age","Weight","Body_Temperature","Heart_Rate"]:
            noise = np.random.normal(0, 0.05, size=len(df))
            noisy[col] = noisy[col].astype(float) * (1 + noise)

        for col in ["Appetite_Loss","Vomiting","Diarrhea","Coughing"]:
            flip = np.random.rand(len(df)) < 0.05
            noisy.loc[flip, col] = 1 - noisy.loc[flip, col]

        augmented.append(noisy)

    return pd.concat(augmented, ignore_index=True)


def generate_healthy(n):
    return pd.DataFrame({
        "Age": np.random.randint(2, 10, n),
        "Weight": np.random.normal(500, 50, n),
        "Body_Temperature": np.random.normal(38.5, 0.3, n),
        "Heart_Rate": np.random.normal(70, 8, n),

        "Appetite_Loss": np.random.choice([0,1], n, p=[0.95, 0.05]),
        "Vomiting": np.random.choice([0,1], n, p=[0.95, 0.05]),
        "Diarrhea": np.random.choice([0,1], n, p=[0.95, 0.05]),
        "Coughing": np.random.choice([0,1], n, p=[0.95, 0.05]),

        "Disease": "healthy"
    })


def build_dataset():
    path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")
    csv_path = path + "/cleaned_animal_disease_prediction.csv"
    df = pd.read_csv(csv_path)

    df = df[df["Animal_Type"] == "Cow"].copy()
    df.rename(columns={'Disease_Prediction':'Disease'}, inplace=True)

    df.replace("Yes", 1, inplace=True)
    df.replace("No", 0, inplace=True)

    df['Body_Temperature'] = df['Body_Temperature'].str.extract(r'(\d+\.?\d*)').astype(float)

    df_sick = augment_data(df, n_times=5)
    df_sick["Disease"] = "sick"

    df_healthy = generate_healthy(len(df_sick))

    df_final = pd.concat([
        df_sick[FEATURES + ["Disease"]],
        df_healthy[FEATURES + ["Disease"]]
    ]).sample(frac=1).reset_index(drop=True)

    return df_final







import os
import pandas as pd
import numpy as np
import kagglehub
from config import FEATURES, DATA_PATH


def augment_data(df: pd.DataFrame, n_times: int = 3) -> pd.DataFrame:
    augmented = []

    for _ in range(n_times):
        noisy = df.copy()

        for col in ["Age", "Weight", "Body_Temperature", "Heart_Rate"]:
            noise = np.random.normal(0, 0.05, size=len(df))
            noisy[col] = noisy[col].astype(float) * (1 + noise)

        for col in ["Appetite_Loss", "Vomiting", "Diarrhea", "Coughing"]:
            flip = np.random.rand(len(df)) < 0.05
            noisy.loc[flip, col] = 1 - noisy.loc[flip, col]

        augmented.append(noisy)

    return pd.concat(augmented, ignore_index=True)


def generate_healthy(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Age": np.random.randint(2, 10, n),
        "Weight": np.random.normal(500, 50, n),
        "Body_Temperature": np.random.normal(38.5, 0.3, n),
        "Heart_Rate": np.random.normal(70, 8, n),
        "Appetite_Loss": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Vomiting": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Diarrhea": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Coughing": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Disease": "healthy",
    })


def build_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")
    csv_path = os.path.join(path, "cleaned_animal_disease_prediction.csv")
    df = pd.read_csv(csv_path)

    df = df[df["Animal_Type"] == "Cow"].copy()
    df.rename(columns={"Disease_Prediction": "Disease"}, inplace=True)

    df.replace("Yes", 1, inplace=True)
    df.replace("No", 0, inplace=True)

    df["Body_Temperature"] = (
        df["Body_Temperature"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    df_sick = augment_data(df, n_times=5)
    df_sick["Disease"] = "sick"

    df_healthy = generate_healthy(len(df_sick))

    df_final = pd.concat(
        [
            df_sick[FEATURES + ["Disease"]],
            df_healthy[FEATURES + ["Disease"]],
        ]
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_final


def main() -> None:
    os.makedirs("data", exist_ok=True)
    df_final = build_dataset()
    df_final.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}")
    print(df_final.head())


if __name__ == "__main__":
    main()
"""

import os
import pandas as pd
import numpy as np
import kagglehub
from config import FEATURES, DATA_PATH


def augment_data(df, n_times=3):
    augmented = []

    for _ in range(n_times):
        noisy = df.copy()

        for col in ["Age", "Weight", "Body_Temperature", "Heart_Rate"]:
            noise = np.random.normal(0, 0.05, size=len(df))
            noisy[col] = noisy[col].astype(float) * (1 + noise)

        for col in ["Appetite_Loss", "Vomiting", "Diarrhea", "Coughing"]:
            flip = np.random.rand(len(df)) < 0.05
            noisy.loc[flip, col] = 1 - noisy.loc[flip, col]

        augmented.append(noisy)

    return pd.concat(augmented, ignore_index=True)


def generate_healthy(n):
    return pd.DataFrame({
        "Age": np.random.randint(2, 10, n),
        "Weight": np.random.normal(500, 50, n),
        "Body_Temperature": np.random.normal(38.5, 0.3, n),
        "Heart_Rate": np.random.normal(70, 8, n),
        "Appetite_Loss": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Vomiting": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Diarrhea": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Coughing": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Disease": "healthy"
    })


def build_dataset():
    path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")
    csv_path = os.path.join(path, "cleaned_animal_disease_prediction.csv")
    df = pd.read_csv(csv_path)

    df = df[df["Animal_Type"] == "Cow"].copy()
    df.rename(columns={"Disease_Prediction": "Disease"}, inplace=True)

    df.replace("Yes", 1, inplace=True)
    df.replace("No", 0, inplace=True)

    df["Body_Temperature"] = (
        df["Body_Temperature"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    df_sick = augment_data(df, n_times=5)
    df_sick["Disease"] = "sick"

    df_healthy = generate_healthy(len(df_sick))

    df_final = pd.concat([
        df_sick[FEATURES + ["Disease"]],
        df_healthy[FEATURES + ["Disease"]]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_final


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df_final = build_dataset()
    df_final.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}")
    print(df_final.head())
