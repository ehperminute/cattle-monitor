
import os
import subprocess
import sys
from config import DATA_PATH, MODEL_PATH


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\nStep failed: {label}")
        sys.exit(result.returncode)


def main() -> None:
    if not os.path.exists(DATA_PATH):
        run_step("Generating dataset", [sys.executable, "generate_data.py"])
    else:
        print(f"\nDataset already exists at {DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        run_step("Training model", [sys.executable, "train.py"])
    else:
        print(f"\nModel already exists at {MODEL_PATH}")

    run_step("Launching demo app", [sys.executable, "app.py"])


if __name__ == "__main__":
    main()
