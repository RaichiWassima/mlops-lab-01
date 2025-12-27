from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "processed.csv"
MODEL_PATH: Final[Path] = ROOT / "models" / "model.joblib"

REPORTS_DIR: Final[Path] = ROOT / "reports"
METRICS_PATH: Final[Path] = REPORTS_DIR / "metrics.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "churn" not in df.columns:
        raise ValueError("Column 'churn' not found in processed.csv")

    X = df.drop(columns=["churn"])
    y = df["churn"].astype(int)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Metrics written to {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
