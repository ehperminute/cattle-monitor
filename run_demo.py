import os
import subprocess
import sys

from config import DATA_PATH, HISTORY_PLOT_PATH, MODEL_PATH, MONITORING_DATA_PATH, SCALER_PATH


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\nStep failed: {label}")
        sys.exit(result.returncode)


def main() -> None:
    if not os.path.exists(DATA_PATH):
        run_step("Generating Base Dataset", [sys.executable, "generate_data.py"])
    else:
        print(f"\nBase dataset already exists at {DATA_PATH}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        run_step("Training Sequential Model", [sys.executable, "train_sequential.py"])
    else:
        print(f"\nSequential model already exists at {MODEL_PATH}")
        if os.path.exists(HISTORY_PLOT_PATH):
            print(f"Training history plot already exists at {HISTORY_PLOT_PATH}")

    if not os.path.exists(MONITORING_DATA_PATH):
        run_step("Generating Monitoring Dataset", [sys.executable, "generate_monitoring_data.py"])
    else:
        print(f"\nMonitoring dataset already exists at {MONITORING_DATA_PATH}")

    run_step("Launching Demo App", [sys.executable, "app.py"])


if __name__ == "__main__":
    main()
