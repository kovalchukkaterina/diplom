# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

sys.dont_write_bytecode = True

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import CANONICAL_FIELD_ORDER, ML_OUTPUT_DIR, detect_mode_csv_files, ensure_directories, extract_windows, read_mode_csv, save_csv


MERGED_DATASET_PATH = ML_OUTPUT_DIR / "gcn_dataset_all_modes.csv"
WINDOWED_DATASET_PATH = ML_OUTPUT_DIR / "gcn_dataset_windowed.csv"


def merge_mode_csv_files() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    csv_files = detect_mode_csv_files()
    if not csv_files:
        raise FileNotFoundError("No mode CSV files were found in data_demo.")

    merged_rows: List[Dict[str, object]] = []
    mode_summaries: List[Dict[str, object]] = []

    for path in csv_files:
        rows = read_mode_csv(path)
        if not rows:
            raise ValueError(f"File has no data: {path}")

        merged_rows.extend(rows)
        mode_summaries.append(
            {
                "source_file": path.name,
                "mode_code": int(rows[0]["mode_code"]),
                "mode_label": str(rows[0]["mode_label"]),
                "row_count": len(rows),
                "t_min": float(rows[0]["t"]),
                "t_max": float(rows[-1]["t"]),
            }
        )

    merged_rows.sort(key=lambda item: (int(item["mode_code"]), float(item["t"])))
    return merged_rows, mode_summaries


def build_windowed_dataset(
    merged_rows: List[Dict[str, object]],
    window_size_sec: float,
    step_size_sec: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in merged_rows:
        grouped.setdefault(str(row["source_file"]), []).append(row)

    window_rows: List[Dict[str, object]] = []
    mode_window_summaries: List[Dict[str, object]] = []

    for source_file, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: float(item["t"]))
        windows, metadata = extract_windows(rows, window_size_sec=window_size_sec, step_size_sec=step_size_sec)
        if not windows:
            raise ValueError(f"Could not build windows for {source_file}")

        window_rows.extend(windows)
        mode_window_summaries.append(metadata)

    window_rows.sort(key=lambda item: (int(item["mode_code"]), int(item["window_index"])))
    return window_rows, mode_window_summaries


def run_preparation(window_size_sec: float = 10.0, step_size_sec: float = 2.0) -> Dict[str, object]:
    ensure_directories()

    merged_rows, mode_summaries = merge_mode_csv_files()
    window_rows, mode_window_summaries = build_windowed_dataset(
        merged_rows=merged_rows,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
    )

    save_csv(merged_rows, fieldnames=CANONICAL_FIELD_ORDER + ["source_file"], path=MERGED_DATASET_PATH)
    save_csv(window_rows, fieldnames=list(window_rows[0].keys()), path=WINDOWED_DATASET_PATH)

    service_fields = {
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
    feature_count = len([name for name in window_rows[0].keys() if name not in service_fields])

    return {
        "window_size_sec_requested": window_size_sec,
        "step_size_sec_requested": step_size_sec,
        "mode_files": mode_summaries,
        "mode_windows": mode_window_summaries,
        "merged_row_count": len(merged_rows),
        "window_row_count": len(window_rows),
        "feature_count": feature_count,
        "merged_dataset_path": str(MERGED_DATASET_PATH),
        "windowed_dataset_path": str(WINDOWED_DATASET_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GCN-195M synthetic data for machine learning.")
    parser.add_argument("--window-size-sec", type=float, default=10.0, help="Sliding window size, s.")
    parser.add_argument("--step-size-sec", type=float, default=2.0, help="Sliding window step, s.")
    args = parser.parse_args()

    summary = run_preparation(window_size_sec=args.window_size_sec, step_size_sec=args.step_size_sec)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
