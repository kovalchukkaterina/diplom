# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import MODEL_DIR, extract_windows, load_csv_rows, read_mode_csv

import joblib
import numpy as np


MODEL_PATH = MODEL_DIR / "gcn_rf_model.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


def load_model_bundle() -> tuple[object, List[str]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено файл моделі: {MODEL_PATH}")
    if not FEATURE_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено список ознак: {FEATURE_NAMES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    return model, feature_names


def rows_to_feature_matrix(rows: List[Dict[str, object]], feature_names: List[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


def predict_from_windowed_csv(path: Path, row_index: int = 0) -> Dict[str, object]:
    model, feature_names = load_model_bundle()
    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"Віконний CSV порожній: {path}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index={row_index} виходить за межі файла {path.name}")

    row = rows[row_index]
    x = rows_to_feature_matrix([row], feature_names)
    predicted_label = str(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    classes = [str(label) for label in model.classes_]

    return {
        "input_type": "windowed_csv",
        "path": str(path),
        "row_index": row_index,
        "predicted_label": predicted_label,
        "probabilities": {label: round(float(prob), 6) for label, prob in zip(classes, probabilities)},
        "window_start_sec": row.get("window_start_sec"),
        "window_end_sec": row.get("window_end_sec"),
    }


def predict_from_raw_csv(path: Path, window_size_sec: float = 10.0, step_size_sec: float = 2.0) -> Dict[str, object]:
    model, feature_names = load_model_bundle()
    rows = read_mode_csv(path)
    windows, metadata = extract_windows(rows, window_size_sec=window_size_sec, step_size_sec=step_size_sec)
    if not windows:
        raise ValueError(f"Не вдалося сформувати вікна для файла {path}")

    x = rows_to_feature_matrix(windows, feature_names)
    predicted_labels = [str(label) for label in model.predict(x)]
    probabilities = model.predict_proba(x)
    mean_probabilities = probabilities.mean(axis=0)
    classes = [str(label) for label in model.classes_]
    majority_label = Counter(predicted_labels).most_common(1)[0][0]

    per_window = []
    for window_row, predicted_label, proba in zip(windows, predicted_labels, probabilities):
        per_window.append(
            {
                "window_index": int(window_row["window_index"]),
                "window_start_sec": float(window_row["window_start_sec"]),
                "window_end_sec": float(window_row["window_end_sec"]),
                "predicted_label": predicted_label,
                "probabilities": {label: round(float(value), 6) for label, value in zip(classes, proba)},
            }
        )

    return {
        "input_type": "raw_csv",
        "path": str(path),
        "window_count": len(windows),
        "majority_label": majority_label,
        "mean_probabilities": {label: round(float(value), 6) for label, value in zip(classes, mean_probabilities)},
        "per_window_predictions": per_window,
        "window_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Інференс режиму роботи ГЦН-195М за збереженою моделлю Random Forest.")
    parser.add_argument("--csv", type=str, help="Шлях до сирого CSV-файла часових рядів.")
    parser.add_argument("--windowed-csv", type=str, help="Шлях до віконного датасету.")
    parser.add_argument("--row-index", type=int, default=0, help="Номер рядка для прогнозу з windowed CSV.")
    parser.add_argument("--window-size-sec", type=float, default=10.0, help="Розмір вікна для сирого CSV, с.")
    parser.add_argument("--step-size-sec", type=float, default=2.0, help="Крок вікна для сирого CSV, с.")
    args = parser.parse_args()

    if args.csv:
        result = predict_from_raw_csv(Path(args.csv), window_size_sec=args.window_size_sec, step_size_sec=args.step_size_sec)
    elif args.windowed_csv:
        result = predict_from_windowed_csv(Path(args.windowed_csv), row_index=args.row_index)
    else:
        parser.error("Потрібно вказати або --csv, або --windowed-csv.")
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
