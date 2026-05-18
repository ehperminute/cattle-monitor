import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from config import FEATURES, TARGET, DATA_PATH, MODEL_PATH, SCALER_PATH, HISTORY_PLOT_PATH


def build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Dense(32, activation="relu", input_shape=(input_dim,)),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run generate_data.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = (df[TARGET] == "sick").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=32,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(X_test_scaled, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["healthy", "sick"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    os.makedirs("model", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=200)
    plt.show()

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
