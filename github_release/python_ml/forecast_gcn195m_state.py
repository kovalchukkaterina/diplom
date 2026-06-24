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

from explain_gcn195m_state import PARAMETER_META
from utils import BASE_SIGNALS, configure_matplotlib

configure_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RISK_RULES_PATH = SCRIPT_DIR / "knowledge_base" / "risk_rules.json"


def load_risk_rules() -> Dict[str, object]:
    if not RISK_RULES_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено rules file: {RISK_RULES_PATH}")
    return json.loads(RISK_RULES_PATH.read_text(encoding="utf-8"))


def compute_signal_slope(times: np.ndarray, values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    shifted_times = times - times[0]
    return float(np.polyfit(shifted_times, values, deg=1)[0])


def harmful_deviation_sigma(parameter_name: str, value: float, normal_mean: float, normal_std: float) -> float:
    harmful_direction = PARAMETER_META[parameter_name]["harmful_direction"]
    if harmful_direction == "high":
        return max((value - normal_mean) / normal_std, 0.0)
    return max((normal_mean - value) / normal_std, 0.0)


def forecast_direction(projected_delta: float, threshold: float) -> str:
    if abs(projected_delta) <= threshold:
        return "стабілізація"
    return "зростання" if projected_delta > 0 else "зниження"


def build_forecast_rows(
    window_rows: Sequence[Dict[str, object]],
    horizon_sec: float,
    step_sec: float,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, np.ndarray]]]:
    observed_times = np.asarray([float(row["t"]) for row in window_rows], dtype=float)
    forecast_offsets = np.arange(step_sec, horizon_sec + 1e-9, step_sec, dtype=float)
    forecast_times = observed_times[-1] + forecast_offsets

    rows: List[Dict[str, object]] = []
    traces: Dict[str, Dict[str, np.ndarray]] = {}
    for row in window_rows:
        rows.append(
            {
                "phase": "observed",
                "time_sec": round(float(row["t"]), 6),
                **{parameter_name: round(float(row[parameter_name]), 6) for parameter_name in BASE_SIGNALS},
            }
        )

    for parameter_name in BASE_SIGNALS:
        observed_values = np.asarray([float(row[parameter_name]) for row in window_rows], dtype=float)
        slope = compute_signal_slope(observed_times, observed_values)
        forecast_values = observed_values[-1] + slope * forecast_offsets
        if observed_values[-1] >= 0.0:
            forecast_values = np.maximum(forecast_values, 0.0)
        traces[parameter_name] = {
            "observed_times": observed_times,
            "observed_values": observed_values,
            "forecast_times": forecast_times,
            "forecast_values": forecast_values,
            "slope": np.asarray([slope], dtype=float),
        }

    for index, time_value in enumerate(forecast_times):
        row = {"phase": "forecast", "time_sec": round(float(time_value), 6)}
        for parameter_name in BASE_SIGNALS:
            row[parameter_name] = round(float(traces[parameter_name]["forecast_values"][index]), 6)
        rows.append(row)

    return rows, traces


def summarize_parameter_forecast(
    parameter_name: str,
    traces: Dict[str, Dict[str, np.ndarray]],
    normal_profile: Dict[str, Dict[str, float]],
    risk_rules: Dict[str, object],
) -> Dict[str, object]:
    signal_trace = traces[parameter_name]
    current_value = float(signal_trace["observed_values"][-1])
    projected_value = float(signal_trace["forecast_values"][-1]) if signal_trace["forecast_values"].size else current_value
    projected_delta = projected_value - current_value
    normal_mean = float(normal_profile[f"{parameter_name}_mean"]["mean"])
    normal_std = float(normal_profile[f"{parameter_name}_mean"]["std"])
    trend_threshold = max(
        float(normal_profile[f"{parameter_name}_std"]["mean"]) * float(risk_rules["forecast"]["trend_eps_scale"]),
        abs(normal_mean) * 0.01,
        1e-6,
    )
    current_sigma = harmful_deviation_sigma(parameter_name, current_value, normal_mean, normal_std)
    projected_sigma = harmful_deviation_sigma(parameter_name, projected_value, normal_mean, normal_std)
    harmful_direction = PARAMETER_META[parameter_name]["harmful_direction"]
    slope = float(signal_trace["slope"][0])
    harmful_trend = slope > 0 if harmful_direction == "high" else slope < 0
    worsening = projected_sigma > current_sigma + 0.15 and harmful_trend
    severe = projected_sigma >= float(risk_rules["forecast"]["severe_deviation_sigma"])
    moderate = projected_sigma >= float(risk_rules["forecast"]["minor_deviation_sigma"])

    return {
        "parameter": parameter_name,
        "display_name": PARAMETER_META[parameter_name]["name"],
        "unit": PARAMETER_META[parameter_name]["unit"],
        "current_value": round(current_value, 6),
        "projected_value": round(projected_value, 6),
        "projected_delta": round(projected_delta, 6),
        "direction": forecast_direction(projected_delta, trend_threshold),
        "current_sigma": round(current_sigma, 6),
        "projected_sigma": round(projected_sigma, 6),
        "harmful_trend": bool(harmful_trend),
        "worsening": bool(worsening),
        "moderate": bool(moderate),
        "severe": bool(severe),
    }


def determine_risk_scenario(
    predicted_label: str,
    class_probability: float,
    parameter_forecasts: Sequence[Dict[str, object]],
    risk_rules: Dict[str, object],
) -> Tuple[str, str]:
    critical_parameters = [item for item in parameter_forecasts if item["severe"] and item["worsening"]]
    worsening_parameters = [item for item in parameter_forecasts if item["moderate"] and item["worsening"]]
    critical_parameter_count = int(risk_rules["forecast"]["critical_parameter_count"])

    if predicted_label == "normal" and not worsening_parameters and not critical_parameters:
        names = ", ".join(item["display_name"] for item in parameter_forecasts[:2])
        return (
            "стабілізація",
            f"За короткостроковим прогнозом істотного віддалення від нормального профілю не очікується; контрольовані параметри ({names}) зберігають стабільну поведінку.",
        )

    if len(critical_parameters) >= critical_parameter_count or (
        len(critical_parameters) >= 1 and class_probability >= 0.85 and len(worsening_parameters) >= 2
    ):
        names = ", ".join(item["display_name"] for item in critical_parameters[:3])
        return (
            "критичний розвиток",
            f"Прогноз показує одночасне поглиблення відхилення за параметрами {names}, що відповідає сценарію критичного розвитку аномального стану.",
        )

    if worsening_parameters:
        names = ", ".join(item["display_name"] for item in worsening_parameters[:3])
        return (
            "погіршення",
            f"Якщо поточний тренд збережеться, очікується подальше погіршення за параметрами {names}; відхилення прогнозовано зростатиме у короткостроковій перспективі.",
        )

    return (
        "стабілізація",
        "Короткостроковий прогноз не показує вираженого наростання аномального відхилення; стан можна оцінити як такий, що локально стабілізується.",
    )


def forecast_state_evolution(
    window_rows: Sequence[Dict[str, object]],
    predicted_label: str,
    class_probability: float,
    normal_profile: Dict[str, Dict[str, float]],
    risk_rules: Dict[str, object],
    horizon_sec: float | None = None,
    step_sec: float | None = None,
) -> Dict[str, object]:
    horizon = float(horizon_sec or risk_rules["forecast"]["horizon_sec_default"])
    step = float(step_sec or risk_rules["forecast"]["step_sec_default"])
    forecast_rows, traces = build_forecast_rows(window_rows, horizon_sec=horizon, step_sec=step)
    parameter_forecasts = [
        summarize_parameter_forecast(parameter_name, traces, normal_profile, risk_rules)
        for parameter_name in BASE_SIGNALS
    ]
    risk_scenario, risk_summary = determine_risk_scenario(
        predicted_label=predicted_label,
        class_probability=class_probability,
        parameter_forecasts=parameter_forecasts,
        risk_rules=risk_rules,
    )

    return {
        "forecast_horizon_sec": horizon,
        "forecast_step_sec": step,
        "forecast_rows": forecast_rows,
        "parameter_forecasts": parameter_forecasts,
        "risk_scenario": risk_scenario,
        "risk_summary": risk_summary,
    }


def save_forecast_plot(forecast_rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    observed_rows = [row for row in forecast_rows if row["phase"] == "observed"]
    future_rows = [row for row in forecast_rows if row["phase"] == "forecast"]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5))
    axes_list = axes.ravel()

    for index, parameter_name in enumerate(BASE_SIGNALS):
        ax = axes_list[index]
        ax.plot(
            [float(row["time_sec"]) for row in observed_rows],
            [float(row[parameter_name]) for row in observed_rows],
            label="спостереження",
            color="tab:blue",
            linewidth=2.2,
        )
        ax.plot(
            [float(row["time_sec"]) for row in future_rows],
            [float(row[parameter_name]) for row in future_rows],
            label="прогноз",
            color="tab:red",
            linestyle="--",
            linewidth=2.2,
        )
        ax.axvline(float(observed_rows[-1]["time_sec"]), color="gray", linestyle=":", linewidth=1.2)
        ax.set_title(PARAMETER_META[parameter_name]["name"], fontsize=12)
        ax.set_xlabel("час, с", fontsize=10)
        ax.set_ylabel(PARAMETER_META[parameter_name]["unit"], fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    summary_ax = axes_list[-1]
    summary_ax.axis("off")
    observed_end = float(observed_rows[-1]["time_sec"])
    forecast_end = float(future_rows[-1]["time_sec"]) if future_rows else observed_end
    summary_ax.text(
        0.05,
        0.82,
        "Короткий підсумок прогнозу",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
    )
    summary_ax.text(
        0.05,
        0.66,
        f"Спостереження до: {observed_end:.1f} с\n"
        f"Горизонт прогнозу: {forecast_end - observed_end:.1f} с\n"
        "Синя лінія — спостереження\nЧервона лінія — короткостроковий прогноз",
        fontsize=11,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F9FC", edgecolor="#4472C4", linewidth=1.2),
    )

    fig.suptitle("Прогнозна оцінка розвитку параметрів стану ГЦН-195М", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_risk_logic_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.axis("off")

    base_style = dict(boxstyle="round,pad=0.5", linewidth=1.4)
    ax.text(
        0.12,
        0.58,
        "Клас режиму\n+\nймовірність моделі",
        ha="center",
        va="center",
        fontsize=12,
        bbox={**base_style, "facecolor": "#E8F1FA", "edgecolor": "#4472C4"},
    )
    ax.text(
        0.40,
        0.58,
        "Відхилення від\nнормального профілю\n(V, Tb, I, H, Q)",
        ha="center",
        va="center",
        fontsize=12,
        bbox={**base_style, "facecolor": "#FCE4D6", "edgecolor": "#C55A11"},
    )
    ax.text(
        0.67,
        0.58,
        "Короткостроковий\nтренд і екстраполяція\nна 40 с",
        ha="center",
        va="center",
        fontsize=12,
        bbox={**base_style, "facecolor": "#FFF2CC", "edgecolor": "#BF9000"},
    )
    ax.text(
        0.90,
        0.58,
        "Сценарій ризику:\nстабілізація /\nпогіршення /\nкритичний розвиток",
        ha="center",
        va="center",
        fontsize=12,
        bbox={**base_style, "facecolor": "#E2F0D9", "edgecolor": "#70AD47"},
    )

    ax.annotate("", xy=(0.27, 0.58), xytext=(0.20, 0.58), arrowprops=dict(arrowstyle="->", linewidth=1.8))
    ax.annotate("", xy=(0.54, 0.58), xytext=(0.47, 0.58), arrowprops=dict(arrowstyle="->", linewidth=1.8))
    ax.annotate("", xy=(0.81, 0.58), xytext=(0.74, 0.58), arrowprops=dict(arrowstyle="->", linewidth=1.8))

    ax.text(
        0.53,
        0.18,
        "Логіка висновку:\n"
        "• якщо параметри близькі до еталону і тренд не погіршується, формується стабілізація\n"
        "• якщо один або кілька домінувальних параметрів віддаляються від еталону, формується погіршення\n"
        "• якщо кілька параметрів одночасно переходять у зону сильного відхилення, формується критичний розвиток",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.suptitle("Схема формування інтегрального висновку про ризик розвитку аномального стану", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
