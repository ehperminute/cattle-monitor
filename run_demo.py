import subprocess
import sys


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\nStep failed: {label}")
        sys.exit(result.returncode)


def main() -> None:
    run_step("Generating base dataset", [sys.executable, "generate_data.py"])
    run_step("Training model", [sys.executable, "train.py"])
    run_step("Generating monitoring dataset", [sys.executable, "generate_monitoring_data.py"])
    run_step("Seeding SQLite database", [sys.executable, "seed_db.py"])
    run_step("Launching demo app", [sys.executable, "app.py"])


if __name__ == "__main__":
    main()
