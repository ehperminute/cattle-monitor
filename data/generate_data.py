import pandas as pd
import numpy as np
import kagglehub
from matplotlib import pyplot as plt
# Download latest version
path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")

print("Path to dataset files:", path)
df = pd.read_csv(path+"/cleaned_animal_disease_prediction.csv")
df_cows = df[df.Animal_Type == "Cow"].copy()
df_cows.rename(columns={'Disease_Prediction':'Disease'}, inplace=True)
df_cows.replace("Yes", 1, inplace=True)
df_cows.replace("No", 0, inplace=True)
df_cows['Body_Temperature'] = df_cows['Body_Temperature'].str.extract(r'(\d+\.?\d*)').astype("float")
df_cows = df_cows.convert_dtypes()
df_cows.columns

features = [
    "Age",
    "Weight",
    "Body_Temperature",
    "Heart_Rate",
    "Appetite_Loss",
    "Vomiting",
    "Diarrhea",
    "Coughing"
]

def augment_data(df, n_times=5):
    augmented = []

    for _ in range(n_times):
        noisy = df.copy()

        # numeric columns
        for col in ["Age", "Weight", "Body_Temperature", "Heart_Rate"]:
            noise = np.random.normal(0, 0.05, size=len(df))
            noisy[col] = noisy[col].astype(float) * (1 + noise)

        # binary symptoms (flip some randomly)
        for col in ["Appetite_Loss", "Vomiting", "Diarrhea", "Coughing"]:
            flip = np.random.rand(len(df)) < 0.05  # 5% flip
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

df_sick_aug = augment_data(df_cows, n_times=5)
df_sick_aug["Disease"] = "sick"
df_healthy = generate_healthy(len(df_sick_aug))

df_final = pd.concat([
    df_sick_aug[features + ["Disease"]],
    df_healthy[features + ["Disease"]]
]).sample(frac=1).reset_index(drop=True)
df_final.groupby("Disease").mean()
df_final[df_final["Disease"]=="sick"]["Body_Temperature"].hist()
df_final[df_final["Disease"]=="healthy"]["Body_Temperature"].hist()
df_final.to_csv("final_data.csv", index=False)
