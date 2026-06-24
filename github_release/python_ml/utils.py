# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_demo"
FIGURES_DIR = PROJECT_ROOT / "figures"
GENERATED_DIR = PROJECT_ROOT / "generated"
ML_OUTPUT_DIR = GENERATED_DIR / "ml"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = DATA_DIR

EXPECTED_MODE_FILES = [
    "gcn_normal.csv",
    "gcn_high_vibration.csv",
    "gcn_bearing_overheat.csv",
    "gcn_head_drop.csv",
    "gcn_motor_overload.csv",
]

MODE_NAME_TO_CODE = {
    "normal": 1,
    "high_vibration": 2,
    "bearing_overheat": 3,
    "head_drop": 4,
    "motor_overload": 5,
}

FIELD_ALIASES: Dict[str, Sequence[str]] = {
    "t": ("t", "time"),
    "V": ("v", "vibration", "v_rms"),
    "Tb": ("tb", "bearing_temperature", "temperature_bearing", "temperature"),
    "I": ("i", "current", "motor_current"),
    "H": ("h", "head", "pressure_head"),
    "Q": ("q", "flow", "flow_rate"),
    "mode_code": ("mode_code", "mode", "class_code"),
    "mode_label": ("mode_label", "label", "mode_name", "class_label"),
}

CANONICAL_FIELD_ORDER = ["t", "mode_code", "mode_label", "V", "Tb", "I", "H", "Q"]
BASE_SIGNALS = ["V", "Tb", "I", "H", "Q"]
FEATURE_SUFFIXES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "first_value",
    "last_value",
    "delta",
    "slope",
    "rms",
]


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    mpl_dir = Path(tempfile.gettempdir()) / "gcn195m_xcos_mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def resolve_field_mapping(fieldnames: Sequence[str]) -> Dict[str, str]:
    normalized = {normalize_name(name): name for name in fieldnames}
    mapping: Dict[str, str] = {}

    for canonical_name, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            alias_key = normalize_name(alias)
            if alias_key in normalized:
                mapping[canonical_name] = normalized[alias_key]
                break

    missing = [name for name in ("t", "V", "Tb", "I", "H", "Q") if name not in mapping]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    return mapping


def detect_mode_csv_files() -> List[Path]:
    files: List[Path] = []
    for filename in EXPECTED_MODE_FILES:
        path = OUTPUT_DIR / filename
        if path.is_file():
            files.append(path)

    if files:
        return files

    for path in sorted(OUTPUT_DIR.glob("gcn_*.csv")):
        if "dataset" in path.name or "summary" in path.name:
            continue
        files.append(path)
    return files


def read_mode_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        mapping = resolve_field_mapping(reader.fieldnames)
        inferred_label = path.stem.replace("gcn_", "")
        inferred_code = MODE_NAME_TO_CODE.get(inferred_label)
        rows: List[Dict[str, object]] = []

        for row in reader:
            if "mode_label" in mapping:
                mode_label = str(row[mapping["mode_label"]]).strip()
            else:
                if inferred_code is None:
                    raise ValueError(f"Cannot infer mode_label from file name: {path.name}")
                mode_label = inferred_label

            if "mode_code" in mapping:
                mode_code = int(float(row[mapping["mode_code"]]))
            else:
                if inferred_code is None:
                    raise ValueError(f"Cannot infer mode_code from file name: {path.name}")
                mode_code = inferred_code

            rows.append(
                {
                    "t": float(row[mapping["t"]]),
                    "mode_code": mode_code,
                    "mode_label": mode_label,
                    "V": float(row[mapping["V"]]),
                    "Tb": float(row[mapping["Tb"]]),
                    "I": float(row[mapping["I"]]),
                    "H": float(row[mapping["H"]]),
                    "Q": float(row[mapping["Q"]]),
                    "source_file": path.name,
                }
            )

    return rows


def infer_dt(times: Sequence[float]) -> float:
    if len(times) < 2:
        raise ValueError("At least two time samples are required to infer dt.")

    diffs = np.diff(np.asarray(times, dtype=float))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise ValueError("Unable to infer dt because time values are not increasing.")
    return float(np.median(positive))


def save_csv(rows: Sequence[Dict[str, object]], fieldnames: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def compute_slope(values: np.ndarray, dt: float) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=float) * dt
    return float(np.polyfit(x, values, deg=1)[0])


def compute_window_features(signal_values: np.ndarray, dt: float) -> Dict[str, float]:
    values = np.asarray(signal_values, dtype=float)
    first_value = float(values[0])
    last_value = float(values[-1])
    min_value = float(np.min(values))
    max_value = float(np.max(values))

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
        "median": float(np.median(values)),
        "first_value": first_value,
        "last_value": last_value,
        "delta": last_value - first_value,
        "slope": compute_slope(values, dt),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
    }


def extract_windows(
    rows: Sequence[Dict[str, object]],
    window_size_sec: float,
    step_size_sec: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not rows:
        return [], {}

    times = [float(row["t"]) for row in rows]
    dt = infer_dt(times)
    window_size_samples = max(2, int(round(window_size_sec / dt)) + 1)
    step_size_samples = max(1, int(round(step_size_sec / dt)))

    total_samples = len(rows)
    mode_code = int(rows[0]["mode_code"])
    mode_label = str(rows[0]["mode_label"])
    source_file = str(rows[0]["source_file"])
    windows: List[Dict[str, object]] = []

    for window_index, start_idx in enumerate(range(0, total_samples - window_size_samples + 1, step_size_samples)):
        end_idx = start_idx + window_size_samples
        window_rows = rows[start_idx:end_idx]
        feature_row: Dict[str, object] = {
            "source_file": source_file,
            "window_index": window_index,
            "window_start_idx": start_idx,
            "window_end_idx": end_idx - 1,
            "window_start_sec": float(window_rows[0]["t"]),
            "window_end_sec": float(window_rows[-1]["t"]),
            "window_size_sec": round((window_size_samples - 1) * dt, 6),
            "step_size_sec": round(step_size_samples * dt, 6),
            "dt": dt,
            "mode_code": mode_code,
            "mode_label": mode_label,
        }

        for signal_name in BASE_SIGNALS:
            signal_values = np.asarray([float(item[signal_name]) for item in window_rows], dtype=float)
            signal_features = compute_window_features(signal_values, dt)
            for suffix, value in signal_features.items():
                feature_row[f"{signal_name}_{suffix}"] = round(value, 6)

        windows.append(feature_row)

    metadata = {
        "dt": dt,
        "window_size_samples": window_size_samples,
        "step_size_samples": step_size_samples,
        "window_count": len(windows),
        "mode_code": mode_code,
        "mode_label": mode_label,
        "source_file": source_file,
    }
    return windows, metadata


def get_feature_columns(fieldnames: Sequence[str]) -> List[str]:
    excluded = {
        "source_file",
        "window_index",
        "window_start_idx",
        "window_end_idx",
        "window_start_sec",
        "window_end_sec",
        "window_size_sec",
        "step_size_sec",
        "dt",
        "mode_code",
        "mode_label",
    }
    return [name for name in fieldnames if name not in excluded]
