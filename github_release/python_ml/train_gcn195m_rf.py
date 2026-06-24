# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    ML_OUTPUT_DIR,
    MODEL_DIR,
    configure_matplotlib,
    ensure_directories,
    get_feature_columns,
    load_csv_rows,
    save_csv,
    write_json,
    write_text,
)

configure_matplotlib()

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


WINDOWED_DATASET_PATH = ML_OUTPUT_DIR / "gcn_dataset_windowed.csv"
CLASSIFICATION_REPORT_TXT = ML_OUTPUT_DIR / "classification_report.txt"
CLASSIFICATION_REPORT_CSV = ML_OUTPUT_DIR / "classification_report.csv"
CONFUSION_MATRIX_PNG = ML_OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PNG = ML_OUTPUT_DIR / "feature_importance.png"
SAMPLE_PREDICTION_PNG = ML_OUTPUT_DIR / "sample_prediction.png"

MODEL_PATH = MODEL_DIR / "gcn_rf_model.joblib"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


def save_classification_report_csv(report_dict: Dict[str, Dict[str, float]]) -> None:
    rows: List[Dict[str, object]] = []
    for label_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            rows.append(
                {
                    "label": label_name,
                    "precision": round(float(metrics.get("precision", 0.0)), 6),
                    "recall": round(float(metrics.get("recall", 0.0)), 6),
                    "f1_score": round(float(metrics.get("f1-score", 0.0)), 6),
                    "support": int(metrics.get("support", 0)),
                }
            )
        else:
            rows.append(
                {
                    "label": label_name,
                    "precision": "",
                    "recall": "",
                    "f1_score": round(float(metrics), 6),
                    "support": "",
                }
            )

    save_csv(rows, fieldnames=["label", "precision", "recall", "f1_score", "support"], path=CLASSIFICATION_REPORT_CSV)


def save_confusion_matrix_plot(cm: np.ndarray, labels: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Матриця помилок класифікації режимів ГЦН-195М", fontsize=13)
    ax.set_xlabel("Передбачений клас", fontsize=11)
    ax.set_ylabel("Істинний клас", fontsize=11)
    ax.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout(pad=1.2)
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=220)
    plt.close(fig)


def save_feature_importance_plot(feature_names: List[str], importances: np.ndarray) -> None:
    ranking = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:15]
    labels = [name for name, _ in reversed(ranking)]
    values = [float(value) for _, value in reversed(ranking)]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(labels, values, color="tab:blue")
    ax.set_xlabel("Відносна важливість", fontsize=11)
    ax.set_title("Найважливіші ознаки моделі Random Forest", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(pad=1.0)
    fig.savefig(FEATURE_IMPORTANCE_PNG, dpi=220)
    plt.close(fig)


def save_sample_prediction_plot(
    test_rows: List[Dict[str, str]],
    y_true: List[str],
    y_pred: List[str],
    y_proba: np.ndarray,
    probability_labels: List[str],
) -> Dict[str, object]:
    sample_index = 0
    best_score = None
    for idx, proba in enumerate(y_proba):
        max_prob = float(np.max(proba))
        score = abs(0.5 - max_prob)
        if best_score is None or score < best_score:
            best_score = score
            sample_index = idx

    sample_row = test_rows[sample_index]
    sample_true = y_true[sample_index]
    sample_pred = y_pred[sample_index]
    sample_proba = y_proba[sample_index]

    probability_pairs = list(zip(probability_labels, [float(item) for item in sample_proba]))
    probability_map = {f"proba_{label}": round(value, 6) for label, value in probability_pairs}

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    bars = ax.bar([label for label, _ in probability_pairs], [value for _, value in probability_pairs], color="tab:green")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Ймовірність", fontsize=11)
    ax.set_title(
        f"Приклад класифікації вікна\nістинний: {sample_true}, передбачений: {sample_pred}",
        fontsize=12,
    )
    ax.tick_params(axis="x", labelrotation=20, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, [value for _, value in probability_pairs]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(pad=1.0)
    fig.savefig(SAMPLE_PREDICTION_PNG, dpi=220)
    plt.close(fig)

    return {
        "test_index": sample_index,
        "source_file": sample_row["source_file"],
        "window_index": sample_row["window_index"],
        "window_start_sec": sample_row["window_start_sec"],
        "window_end_sec": sample_row["window_end_sec"],
        "true_label": sample_true,
        "predicted_label": sample_pred,
        **probability_map,
    }


def train_random_forest(random_state: int = 42, test_size: float = 0.2) -> Dict[str, object]:
    ensure_directories()
    if not WINDOWED_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено віконний датасет: {WINDOWED_DATASET_PATH}")

    window_rows = load_csv_rows(WINDOWED_DATASET_PATH)
    if not window_rows:
        raise ValueError("Віконний датасет порожній.")

    feature_columns = get_feature_columns(window_rows[0].keys())
    x = np.asarray([[float(row[column]) for column in feature_columns] for row in window_rows], dtype=float)
    y_labels = np.asarray([str(row["mode_label"]) for row in window_rows])

    label_to_code: Dict[str, int] = {}
    for row in window_rows:
        label_to_code[str(row["mode_label"])] = int(float(row["mode_code"]))
    label_order = [label for label, _ in sorted(label_to_code.items(), key=lambda item: item[1])]

    row_indices = np.arange(len(window_rows))
    x_train, x_test, y_train, y_test, _, idx_test = train_test_split(
        x,
        y_labels,
        row_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels,
    )

    estimator = RandomForestClassifier(random_state=random_state, n_jobs=1)
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=1,
        cv=cv,
        refit=True,
    )
    grid.fit(x_train, y_train)

    model: RandomForestClassifier = grid.best_estimator_
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    probability_labels = [str(label) for label in model.classes_]

    accuracy = float(accuracy_score(y_test, y_pred))
    report_text = classification_report(y_test, y_pred, digits=4)
    report_dict = classification_report(y_test, y_pred, output_dict=True, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=label_order)

    write_text(CLASSIFICATION_REPORT_TXT, report_text)
    save_classification_report_csv(report_dict)
    save_confusion_matrix_plot(cm, labels=label_order)
    save_feature_importance_plot(feature_columns, model.feature_importances_)

    test_rows = [window_rows[int(idx)] for idx in idx_test]
    sample_prediction = save_sample_prediction_plot(
        test_rows=test_rows,
        y_true=list(y_test),
        y_pred=list(y_pred),
        y_proba=y_proba,
        probability_labels=probability_labels,
    )

    joblib.dump(model, MODEL_PATH)
    write_json(
        LABEL_MAPPING_PATH,
        {
            "label_to_code": label_to_code,
            "code_to_label": {str(code): label for label, code in label_to_code.items()},
        },
    )
    write_json(FEATURE_NAMES_PATH, feature_columns)

    return {
        "accuracy": accuracy,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "feature_count": int(x.shape[1]),
        "class_count": len(label_order),
        "labels": label_order,
        "best_params": grid.best_params_,
        "best_cv_accuracy": float(grid.best_score_),
        "windowed_dataset_path": str(WINDOWED_DATASET_PATH),
        "model_path": str(MODEL_PATH),
        "classification_report_txt": str(CLASSIFICATION_REPORT_TXT),
        "classification_report_csv": str(CLASSIFICATION_REPORT_CSV),
        "confusion_matrix_png": str(CONFUSION_MATRIX_PNG),
        "feature_importance_png": str(FEATURE_IMPORTANCE_PNG),
        "sample_prediction_png": str(SAMPLE_PREDICTION_PNG),
        "sample_prediction": sample_prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Навчання Random Forest для класифікації режимів ГЦН-195М.")
    parser.add_argument("--random-state", type=int, default=42, help="Початкове значення генератора випадкових чисел.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Частка тестової вибірки.")
    args = parser.parse_args()

    summary = train_random_forest(random_state=args.random_state, test_size=args.test_size)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
