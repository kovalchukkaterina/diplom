# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import BASE_SIGNALS, FEATURE_SUFFIXES, ML_OUTPUT_DIR, MODEL_DIR, get_feature_columns, load_csv_rows

import joblib
import numpy as np


WINDOWED_DATASET_PATH = ML_OUTPUT_DIR / "gcn_dataset_windowed.csv"
MODEL_PATH = MODEL_DIR / "gcn_rf_model.joblib"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
ANOMALY_CAUSES_PATH = SCRIPT_DIR / "knowledge_base" / "anomaly_causes.json"

PARAMETER_META: Dict[str, Dict[str, str]] = {
    "V": {"name": "вібрація", "unit": "мм/с", "harmful_direction": "high"},
    "Tb": {"name": "температура підшипникового вузла", "unit": "°C", "harmful_direction": "high"},
    "I": {"name": "струм електродвигуна", "unit": "А", "harmful_direction": "high"},
    "H": {"name": "напір", "unit": "кгс/см²", "harmful_direction": "low"},
    "Q": {"name": "витрата", "unit": "м³/год", "harmful_direction": "low"},
}


def load_model_artifacts() -> Tuple[object, List[str], Dict[str, object], Dict[str, float]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено файл моделі: {MODEL_PATH}")
    if not LABEL_MAPPING_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено mapping класів: {LABEL_MAPPING_PATH}")
    if not FEATURE_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено список ознак: {FEATURE_NAMES_PATH}")

    model = joblib.load(MODEL_PATH)
    label_mapping = json.loads(LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    importance_map = {
        feature_name: float(importance)
        for feature_name, importance in zip(feature_names, model.feature_importances_)
    }
    return model, feature_names, label_mapping, importance_map


def load_anomaly_knowledge_base() -> Dict[str, object]:
    if not ANOMALY_CAUSES_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено knowledge base: {ANOMALY_CAUSES_PATH}")
    return json.loads(ANOMALY_CAUSES_PATH.read_text(encoding="utf-8"))


def build_normal_reference_profile(path: Path = WINDOWED_DATASET_PATH) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        from prepare_gcn195m_ml_data import run_preparation

        run_preparation()

    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"Віконний датасет порожній: {path}")

    normal_rows = [row for row in rows if str(row["mode_label"]) == "normal"]
    if not normal_rows:
        raise ValueError("У віконному датасеті відсутній нормальний клас для побудови еталонного профілю.")

    feature_columns = get_feature_columns(normal_rows[0].keys())
    profile: Dict[str, Dict[str, float]] = {}
    for feature_name in feature_columns:
        values = np.asarray([float(row[feature_name]) for row in normal_rows], dtype=float)
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        if std_value <= 1e-9:
            std_value = max(abs(mean_value) * 0.01, 1e-6)
        profile[feature_name] = {"mean": mean_value, "std": std_value}
    return profile


def z_score(value: float, reference_mean: float, reference_std: float) -> float:
    return (value - reference_mean) / reference_std


def determine_level_state(parameter_name: str, mean_value: float, normal_value: float, normal_std: float) -> str:
    z = z_score(mean_value, normal_value, normal_std)
    if z >= 1.0:
        return "high"
    if z <= -1.0:
        return "low"
    return "normal"


def determine_trend_state(parameter_name: str, delta_value: float, normal_std_feature: float) -> str:
    threshold = max(normal_std_feature * 0.25, 1e-6)
    if abs(delta_value) <= threshold:
        return "stable"
    return "rising" if delta_value > 0 else "falling"


def harmful_feature_component(parameter_name: str, suffix: str, z_value: float) -> float:
    harmful_direction = PARAMETER_META[parameter_name]["harmful_direction"]
    if suffix in {"std", "range"}:
        return max(z_value, 0.0)

    sign = 1.0 if harmful_direction == "high" else -1.0
    return max(sign * z_value, 0.0)


def parameter_direction_text(level_state: str, trend_state: str, parameter_name: str) -> str:
    display_name = PARAMETER_META[parameter_name]["name"]
    if level_state == "high" and trend_state == "rising":
        return f"{display_name} є підвищеною і зберігає зростальний тренд"
    if level_state == "high" and trend_state == "stable":
        return f"{display_name} утримується на підвищеному рівні"
    if level_state == "high" and trend_state == "falling":
        return f"{display_name} залишається підвищеною, але локально знижується"
    if level_state == "low" and trend_state == "falling":
        return f"{display_name} є зниженою і продовжує спадати"
    if level_state == "low" and trend_state == "stable":
        return f"{display_name} утримується на зниженому рівні"
    if level_state == "low" and trend_state == "rising":
        return f"{display_name} залишається зниженою, але локально відновлюється"
    if trend_state == "rising":
        return f"{display_name} близька до норми, однак має зростальний тренд"
    if trend_state == "falling":
        return f"{display_name} близька до норми, однак має спадний тренд"
    return f"{display_name} залишається поблизу типового рівня"


def analyze_parameter(
    parameter_name: str,
    window_row: Dict[str, object],
    normal_profile: Dict[str, Dict[str, float]],
    importance_map: Dict[str, float],
) -> Dict[str, object]:
    mean_value = float(window_row[f"{parameter_name}_mean"])
    normal_mean = normal_profile[f"{parameter_name}_mean"]["mean"]
    normal_std = normal_profile[f"{parameter_name}_mean"]["std"]
    delta_value = float(window_row[f"{parameter_name}_delta"])
    slope_value = float(window_row[f"{parameter_name}_slope"])
    window_std = float(window_row[f"{parameter_name}_std"])
    normal_window_std = normal_profile[f"{parameter_name}_std"]["mean"]
    level_state = determine_level_state(parameter_name, mean_value, normal_mean, normal_std)
    trend_state = determine_trend_state(parameter_name, delta_value, normal_window_std)

    feature_breakdown: List[Dict[str, object]] = []
    for suffix in FEATURE_SUFFIXES:
        feature_name = f"{parameter_name}_{suffix}"
        value = float(window_row[feature_name])
        reference = normal_profile[feature_name]
        z_value = z_score(value, reference["mean"], reference["std"])
        harmful_component = harmful_feature_component(parameter_name, suffix, z_value)
        importance_weight = 1.0 + 8.0 * float(importance_map.get(feature_name, 0.0))
        weighted_score = harmful_component * importance_weight
        feature_breakdown.append(
            {
                "feature_name": feature_name,
                "suffix": suffix,
                "value": round(value, 6),
                "normal_mean": round(reference["mean"], 6),
                "z_score": round(z_value, 6),
                "importance": round(float(importance_map.get(feature_name, 0.0)), 6),
                "weighted_score": round(weighted_score, 6),
            }
        )

    feature_breakdown.sort(key=lambda item: float(item["weighted_score"]), reverse=True)
    dominant_support = feature_breakdown[:3]
    parameter_score = sum(float(item["weighted_score"]) for item in dominant_support)
    relative_diff_percent = 0.0 if abs(normal_mean) < 1e-9 else ((mean_value - normal_mean) / abs(normal_mean)) * 100.0

    return {
        "parameter": parameter_name,
        "display_name": PARAMETER_META[parameter_name]["name"],
        "unit": PARAMETER_META[parameter_name]["unit"],
        "harmful_direction": PARAMETER_META[parameter_name]["harmful_direction"],
        "mean_value": round(mean_value, 6),
        "normal_mean": round(normal_mean, 6),
        "relative_diff_percent": round(relative_diff_percent, 4),
        "window_std": round(window_std, 6),
        "normal_window_std": round(normal_window_std, 6),
        "delta": round(delta_value, 6),
        "slope": round(slope_value, 6),
        "level_state": level_state,
        "trend_state": trend_state,
        "parameter_score": round(parameter_score, 6),
        "feature_breakdown": feature_breakdown,
        "summary_text": parameter_direction_text(level_state, trend_state, parameter_name),
    }


def evaluate_trigger(parameter_summary: Dict[str, object], trigger_state: str) -> float:
    level_state = str(parameter_summary["level_state"])
    trend_state = str(parameter_summary["trend_state"])
    if trigger_state in {"high", "low", "normal"}:
        return 1.0 if level_state == trigger_state else 0.0
    if trigger_state in {"rising", "falling", "stable"}:
        return 1.0 if trend_state == trigger_state else 0.0
    return 0.0


def rank_probable_causes(
    predicted_label: str,
    parameter_summaries: Dict[str, Dict[str, object]],
    dominant_parameters: Sequence[str],
    knowledge_base: Dict[str, object],
) -> List[Dict[str, object]]:
    class_entry = knowledge_base[predicted_label]
    causes = class_entry.get("causes", [])
    ranked: List[Dict[str, object]] = []

    for cause in causes:
        triggers = cause.get("triggers", [])
        trigger_scores: List[float] = []
        matched_triggers: List[str] = []
        for trigger in triggers:
            parameter_name = str(trigger["parameter"])
            trigger_state = str(trigger["state"])
            score = evaluate_trigger(parameter_summaries[parameter_name], trigger_state)
            trigger_scores.append(score)
            if score >= 1.0:
                matched_triggers.append(f"{parameter_name}:{trigger_state}")

        match_ratio = float(sum(trigger_scores) / len(trigger_scores)) if trigger_scores else 0.0
        dominant_bonus = 0.2 if any(str(trigger["parameter"]) in dominant_parameters for trigger in triggers) else 0.0
        priority = int(cause.get("priority", 10))
        priority_bonus = max(0.0, 0.3 - (priority - 1) * 0.05)
        score_total = match_ratio + dominant_bonus + priority_bonus
        if predicted_label == "normal" or match_ratio > 0.0:
            ranked.append(
                {
                    "name": cause["name"],
                    "priority": priority,
                    "score": round(score_total, 6),
                    "matched_triggers": matched_triggers,
                    "rationale": cause["rationale"],
                    "justification": (
                        cause["rationale"]
                        if matched_triggers
                        else "Причина включена як базове пояснення режиму без виражених суперечностей із поточними ознаками."
                    ),
                }
            )

    ranked.sort(key=lambda item: (float(item["score"]), -int(item["priority"])), reverse=True)
    return ranked[:4]


def build_key_feature_rows(parameter_summaries: Dict[str, Dict[str, object]], top_n: int = 6) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for parameter_summary in parameter_summaries.values():
        rows.extend(parameter_summary["feature_breakdown"])
    rows.sort(key=lambda item: float(item["weighted_score"]), reverse=True)
    return rows[:top_n]


def build_explanation_text(
    predicted_label: str,
    probability: float,
    parameter_summaries: Dict[str, Dict[str, object]],
    dominant_parameters: Sequence[str],
    probable_causes: Sequence[Dict[str, object]],
    class_description: str,
) -> str:
    if predicted_label == "normal":
        dominant_fragment = ", ".join(parameter_summaries[param]["summary_text"] for param in dominant_parameters[:2])
        return (
            f"Стан віднесено до класу '{predicted_label}' з імовірністю {probability:.3f}. "
            f"Поточне вікно відповідає опису '{class_description.lower()}'. "
            f"Найбільші відхилення залишаються помірними: {dominant_fragment}. "
            f"Ознак стійкого розвитку аномалії за ключовими параметрами не виявлено."
        )

    dominant_fragment = "; ".join(parameter_summaries[param]["summary_text"] for param in dominant_parameters[:3])
    cause_fragment = probable_causes[0]["name"] if probable_causes else "аномального стану, узгодженого з поточними ознаками"
    return (
        f"Стан віднесено до класу '{predicted_label}' з імовірністю {probability:.3f}. "
        f"Це узгоджується з описом режиму: {class_description.lower()}. "
        f"Найбільші відхилення сформували такі параметри: {dominant_fragment}. "
        f"Найімовірніше пояснення відповідає причині '{cause_fragment}'."
    )


def explain_window_state(
    window_row: Dict[str, object],
    predicted_label: str,
    probability_map: Dict[str, float],
    normal_profile: Dict[str, Dict[str, float]],
    importance_map: Dict[str, float],
    knowledge_base: Dict[str, object],
) -> Dict[str, object]:
    parameter_summaries = {
        parameter_name: analyze_parameter(parameter_name, window_row, normal_profile, importance_map)
        for parameter_name in BASE_SIGNALS
    }
    dominant_parameters = [
        item["parameter"]
        for item in sorted(
            parameter_summaries.values(),
            key=lambda summary: float(summary["parameter_score"]),
            reverse=True,
        )
    ]
    key_features = build_key_feature_rows(parameter_summaries)
    probable_causes = rank_probable_causes(predicted_label, parameter_summaries, dominant_parameters, knowledge_base)
    class_description = str(knowledge_base[predicted_label]["description"])
    class_probability = float(probability_map[predicted_label])
    explanation_text = build_explanation_text(
        predicted_label=predicted_label,
        probability=class_probability,
        parameter_summaries=parameter_summaries,
        dominant_parameters=dominant_parameters,
        probable_causes=probable_causes,
        class_description=class_description,
    )

    recommended_actions = knowledge_base[predicted_label].get("recommended_actions", [])
    integral_conclusion = (
        "Найбільший вклад у рішення моделі внесли параметри "
        + ", ".join(parameter_summaries[param]["display_name"] for param in dominant_parameters[:3])
        + "."
    )

    return {
        "predicted_label": predicted_label,
        "class_probability": round(class_probability, 6),
        "class_description": class_description,
        "dominant_parameters": dominant_parameters[:3],
        "parameter_summaries": list(parameter_summaries.values()),
        "key_features": key_features,
        "probable_causes": probable_causes,
        "recommended_actions": recommended_actions,
        "explanation_text": explanation_text,
        "integral_conclusion": integral_conclusion,
    }
