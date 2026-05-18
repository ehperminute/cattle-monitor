FEATURES = [
    "age",
    "weight",
    "body_temperature",
    "heart_rate",
    "appetite_loss",
    "vomiting",
    "diarrhea",
    "coughing",
]

TARGET = "disease"

DATA_PATH = "data/final_data.csv"
MONITORING_DATA_PATH = "data/monitoring_data.csv"
MODEL_PATH = "model/sequential_classifier.keras"
SCALER_PATH = "model/feature_scaler.joblib"
HISTORY_PLOT_PATH = "artifacts/train_history.png"

SUPPORTED_LANGS = ["es", "en", "nawatlahtolli", "hnahnu", "tuun_savi"]
DEFAULT_LANG = "es"
