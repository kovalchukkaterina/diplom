# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from explain_gcn195m_state import (
    PARAMETER_META,
    build_normal_reference_profile,
    explain_window_state,
    load_anomaly_knowledge_base,
    load_model_artifacts,
)
from forecast_gcn195m_state import forecast_state_evolution, load_risk_rules, save_forecast_plot, save_risk_logic_figure
from utils import DATA_DIR, GENERATED_DIR, configure_matplotlib, ensure_directories, extract_windows, read_mode_csv, save_csv, write_json, write_text

configure_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPLANATION_OUTPUT_DIR = GENERATED_DIR / "explanation_forecast"
SAMPLE_EXPLANATION_JSON = EXPLANATION_OUTPUT_DIR / "sample_explanation.json"
SAMPLE_EXPLANATION_TXT = EXPLANATION_OUTPUT_DIR / "sample_explanation.txt"
SAMPLE_CAUSES_CSV = EXPLANATION_OUTPUT_DIR / "sample_causes.csv"
SAMPLE_FORECAST_CSV = EXPLANATION_OUTPUT_DIR / "sample_forecast.csv"
SAMPLE_FORECAST_PNG = EXPLANATION_OUTPUT_DIR / "sample_forecast_plot.png"
SAMPLE_RISK_SUMMARY_TXT = EXPLANATION_OUTPUT_DIR / "sample_risk_summary.txt"
SAMPLE_EXPLANATION_CARD_PNG = EXPLANATION_OUTPUT_DIR / "sample_explanation_card.png"
RISK_LOGIC_SCHEME_PNG = EXPLANATION_OUTPUT_DIR / "risk_logic_scheme.png"


def rows_to_feature_matrix(rows: Sequence[Dict[str, object]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


def probability_map_for_row(model: object, feature_names: Sequence[str], row: Dict[str, object]) -> Tuple[str, Dict[str, float]]:
    x = rows_to_feature_matrix([row], feature_names)
    predicted_label = str(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    classes = [str(label) for label in model.classes_]
    probability_map = {label: round(float(prob), 6) for label, prob in zip(classes, probabilities)}
    return predicted_label, probability_map


def find_selected_window_from_raw_csv(
    path: Path,
    model: object,
    feature_names: Sequence[str],
    window_size_sec: float,
    step_size_sec: float,
) -> Dict[str, object]:
    rows = read_mode_csv(path)
    windows, metadata = extract_windows(rows, window_size_sec=window_size_sec, step_size_sec=step_size_sec)
    if not windows:
        raise ValueError(f"Не вдалося сформувати вікна для файла {path}")

    x = rows_to_feature_matrix(windows, feature_names)
    predicted_labels = [str(label) for label in model.predict(x)]
    probabilities = model.predict_proba(x)
    classes = [str(label) for label in model.classes_]
    majority_label = Counter(predicted_labels).most_common(1)[0][0]
    class_index = classes.index(majority_label)
    candidate_indices = [index for index, label in enumerate(predicted_labels) if label == majority_label]
    selected_index = max(candidate_indices, key=lambda index: (probabilities[index][class_index], index))
    selected_window = windows[selected_index]
    start_idx = int(selected_window["window_start_idx"])
    end_idx = int(selected_window["window_end_idx"])

    probability_map = {label: round(float(probabilities[selected_index][idx]), 6) for idx, label in enumerate(classes)}
    return {
        "input_type": "raw_csv",
        "source_path": str(path),
        "window_metadata": metadata,
        "selected_window": selected_window,
        "selected_window_rows": rows[start_idx : end_idx + 1],
        "selected_index": selected_index,
        "predicted_label": majority_label,
        "probability_map": probability_map,
    }


def reconstruct_window_rows_from_windowed_input(window_row: Dict[str, object]) -> List[Dict[str, object]]:
    source_path = DATA_DIR / str(window_row["source_file"])
    source_rows = read_mode_csv(source_path)
    start_idx = int(float(window_row["window_start_idx"]))
    end_idx = int(float(window_row["window_end_idx"]))
    return source_rows[start_idx : end_idx + 1]


def find_selected_window_from_windowed_csv(
    path: Path,
    row_index: int,
    model: object,
    feature_names: Sequence[str],
) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Віконний CSV порожній: {path}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index={row_index} виходить за межі файла {path.name}")

    selected_window = rows[row_index]
    predicted_label, probability_map = probability_map_for_row(model, feature_names, selected_window)

    return {
        "input_type": "windowed_csv",
        "source_path": str(path),
        "selected_window": selected_window,
        "selected_window_rows": reconstruct_window_rows_from_windowed_input(selected_window),
        "selected_index": row_index,
        "predicted_label": predicted_label,
        "probability_map": probability_map,
    }


def format_probability(probability: float) -> str:
    return f"{probability * 100.0:.1f}%"


def build_explanation_text_block(explanation_result: Dict[str, object]) -> str:
    lines = [
        f"Визначений клас: {explanation_result['predicted_label']}",
        f"Ймовірність класу: {format_probability(float(explanation_result['class_probability']))}",
        "Домінувальні параметри: "
        + ", ".join(PARAMETER_META[param]["name"] for param in explanation_result["dominant_parameters"]),
        explanation_result["explanation_text"],
    ]
    if explanation_result["probable_causes"]:
        lines.append("Ймовірні причини:")
        for cause in explanation_result["probable_causes"][:3]:
            lines.append(f"- {cause['name']}: {cause['justification']}")
    return "\n".join(lines)


def save_explanation_card(
    explanation_result: Dict[str, object],
    forecast_result: Dict[str, object],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 7.8))
    ax.axis("off")

    causes = explanation_result["probable_causes"][:3]
    cause_lines = "\n".join(f"• {cause['name']}" for cause in causes) if causes else "• виражені причини не виявлені"
    dominant_lines = "\n".join(f"• {PARAMETER_META[param]['name']}" for param in explanation_result["dominant_parameters"])
    label_display = {
        "normal": "нормальний режим",
        "high_vibration": "підвищена вібрація",
        "bearing_overheat": "перегрів підшипникового вузла",
        "head_drop": "зниження напору",
        "motor_overload": "перевантаження електродвигуна",
    }
    label_text = label_display.get(explanation_result["predicted_label"], explanation_result["predicted_label"])
    text = (
        f"Клас: {label_text} ({explanation_result['predicted_label']})\n"
        f"Ймовірність: {format_probability(float(explanation_result['class_probability']))}\n\n"
        f"Домінувальні параметри:\n{dominant_lines}\n\n"
        f"Ймовірні причини:\n{cause_lines}\n\n"
        f"Сценарій ризику: {forecast_result['risk_scenario']}\n"
        f"{forecast_result['risk_summary']}"
    )

    ax.text(
        0.02,
        0.98,
        "Приклад формування пояснення причин аномального стану ГЦН-195М",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.87,
        text,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.95", facecolor="#F7F9FC", edgecolor="#4472C4", linewidth=1.5),
    )
    fig.tight_layout(pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_supporting_files(
    explanation_result: Dict[str, object],
    forecast_result: Dict[str, object],
    sample_result: Dict[str, object],
) -> None:
    ensure_directories()
    EXPLANATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_json(SAMPLE_EXPLANATION_JSON, sample_result)
    write_text(SAMPLE_EXPLANATION_TXT, build_explanation_text_block(explanation_result))
    write_text(SAMPLE_RISK_SUMMARY_TXT, forecast_result["risk_summary"])

    save_csv(
        rows=explanation_result["probable_causes"],
        fieldnames=["name", "priority", "score", "matched_triggers", "rationale", "justification"],
        path=SAMPLE_CAUSES_CSV,
    )
    save_csv(
        rows=forecast_result["forecast_rows"],
        fieldnames=["phase", "time_sec", "V", "Tb", "I", "H", "Q"],
        path=SAMPLE_FORECAST_CSV,
    )
    save_explanation_card(explanation_result, forecast_result, SAMPLE_EXPLANATION_CARD_PNG)
    save_forecast_plot(forecast_result["forecast_rows"], SAMPLE_FORECAST_PNG)
    save_risk_logic_figure(RISK_LOGIC_SCHEME_PNG)


def assemble_sample_result(
    selection_result: Dict[str, object],
    explanation_result: Dict[str, object],
    forecast_result: Dict[str, object],
) -> Dict[str, object]:
    selected_window = selection_result["selected_window"]
    return {
        "input_type": selection_result["input_type"],
        "input_source": selection_result["source_path"],
        "selected_window_index": int(selection_result["selected_index"]),
        "window_start_sec": float(selected_window["window_start_sec"]),
        "window_end_sec": float(selected_window["window_end_sec"]),
        "predicted_label": explanation_result["predicted_label"],
        "class_probability": explanation_result["class_probability"],
        "probability_map": selection_result["probability_map"],
        "dominant_parameters": explanation_result["dominant_parameters"],
        "key_features": explanation_result["key_features"],
        "probable_causes": explanation_result["probable_causes"],
        "explanation_text": explanation_result["explanation_text"],
        "integral_conclusion": explanation_result["integral_conclusion"],
        "risk_scenario": forecast_result["risk_scenario"],
        "risk_summary": forecast_result["risk_summary"],
        "parameter_forecasts": forecast_result["parameter_forecasts"],
    }


def run_explanation_and_forecast(
    csv_path: Path | None = None,
    windowed_csv_path: Path | None = None,
    row_index: int = 0,
    window_size_sec: float = 10.0,
    step_size_sec: float = 2.0,
    forecast_horizon_sec: float | None = None,
    forecast_step_sec: float | None = None,
) -> Dict[str, object]:
    ensure_directories()
    EXPLANATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, feature_names, _, importance_map = load_model_artifacts()
    normal_profile = build_normal_reference_profile()
    anomaly_kb = load_anomaly_knowledge_base()
    risk_rules = load_risk_rules()

    if csv_path:
        selection_result = find_selected_window_from_raw_csv(
            path=csv_path,
            model=model,
            feature_names=feature_names,
            window_size_sec=window_size_sec,
            step_size_sec=step_size_sec,
        )
    elif windowed_csv_path:
        selection_result = find_selected_window_from_windowed_csv(
            path=windowed_csv_path,
            row_index=row_index,
            model=model,
            feature_names=feature_names,
        )
    else:
        raise ValueError("Потрібно вказати csv_path або windowed_csv_path.")

    explanation_result = explain_window_state(
        window_row=selection_result["selected_window"],
        predicted_label=selection_result["predicted_label"],
        probability_map=selection_result["probability_map"],
        normal_profile=normal_profile,
        importance_map=importance_map,
        knowledge_base=anomaly_kb,
    )
    forecast_result = forecast_state_evolution(
        window_rows=selection_result["selected_window_rows"],
        predicted_label=explanation_result["predicted_label"],
        class_probability=float(explanation_result["class_probability"]),
        normal_profile=normal_profile,
        risk_rules=risk_rules,
        horizon_sec=forecast_horizon_sec,
        step_sec=forecast_step_sec,
    )

    sample_result = assemble_sample_result(selection_result, explanation_result, forecast_result)
    save_supporting_files(
        explanation_result=explanation_result,
        forecast_result=forecast_result,
        sample_result=sample_result,
    )

    return {
        "sample_explanation_json": str(SAMPLE_EXPLANATION_JSON),
        "sample_explanation_txt": str(SAMPLE_EXPLANATION_TXT),
        "sample_causes_csv": str(SAMPLE_CAUSES_CSV),
        "sample_forecast_csv": str(SAMPLE_FORECAST_CSV),
        "sample_forecast_plot_png": str(SAMPLE_FORECAST_PNG),
        "sample_risk_summary_txt": str(SAMPLE_RISK_SUMMARY_TXT),
        "sample_explanation_card_png": str(SAMPLE_EXPLANATION_CARD_PNG),
        "risk_logic_scheme_png": str(RISK_LOGIC_SCHEME_PNG),
        "predicted_label": sample_result["predicted_label"],
        "class_probability": sample_result["class_probability"],
        "risk_scenario": sample_result["risk_scenario"],
        "window_start_sec": sample_result["window_start_sec"],
        "window_end_sec": sample_result["window_end_sec"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Пояснення причин аномалії та прогнозна оцінка розвитку стану ГЦН-195М.")
    parser.add_argument("--csv", type=str, help="Шлях до сирого CSV-файла часових рядів.")
    parser.add_argument("--windowed-csv", type=str, help="Шлях до віконного CSV-файла ознак.")
    parser.add_argument("--row-index", type=int, default=0, help="Номер рядка для аналізу з windowed CSV.")
    parser.add_argument("--window-size-sec", type=float, default=10.0, help="Розмір ковзного вікна для сирого CSV, с.")
    parser.add_argument("--step-size-sec", type=float, default=2.0, help="Крок зсуву вікна для сирого CSV, с.")
    parser.add_argument("--forecast-horizon-sec", type=float, default=40.0, help="Горизонт короткострокового прогнозу, с.")
    parser.add_argument("--forecast-step-sec", type=float, default=2.0, help="Крок прогнозного ряду, с.")
    args = parser.parse_args()

    if not args.csv and not args.windowed_csv:
        parser.error("Потрібно вказати --csv або --windowed-csv.")

    summary = run_explanation_and_forecast(
        csv_path=Path(args.csv) if args.csv else None,
        windowed_csv_path=Path(args.windowed_csv) if args.windowed_csv else None,
        row_index=args.row_index,
        window_size_sec=args.window_size_sec,
        step_size_sec=args.step_size_sec,
        forecast_horizon_sec=args.forecast_horizon_sec,
        forecast_step_sec=args.forecast_step_sec,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
