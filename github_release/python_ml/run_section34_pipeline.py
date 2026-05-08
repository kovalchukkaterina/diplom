# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_gcn195m_ml_data import run_preparation
from train_gcn195m_rf import train_random_forest
from utils import ML_OUTPUT_DIR, configure_matplotlib, ensure_directories

configure_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOW_FEATURE_SCHEME_PNG = ML_OUTPUT_DIR / "window_feature_scheme.png"


def build_window_feature_scheme(window_size_sec: float, step_size_sec: float, feature_count: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    box_style_in = dict(boxstyle="round,pad=0.5", facecolor="#E8F1FA", edgecolor="#4472C4", linewidth=1.4)
    box_style_mid = dict(boxstyle="round,pad=0.5", facecolor="#FCE4D6", edgecolor="#C55A11", linewidth=1.4)
    box_style_out = dict(boxstyle="round,pad=0.5", facecolor="#E2F0D9", edgecolor="#70AD47", linewidth=1.4)

    ax.text(
        0.15,
        0.55,
        f"Sliding window over raw signals\nV, Tb, I, H, Q\nlength {window_size_sec:.1f} s\nstep {step_size_sec:.1f} s",
        ha="center",
        va="center",
        fontsize=12,
        bbox=box_style_in,
    )
    ax.text(
        0.50,
        0.55,
        "Per-parameter features\nmean, std, min, max, range,\nmedian, first_value, last_value,\ndelta, slope, rms",
        ha="center",
        va="center",
        fontsize=12,
        bbox=box_style_mid,
    )
    ax.text(
        0.84,
        0.55,
        f"Feature vector for one window\n{feature_count} features\n+ mode_code, mode_label",
        ha="center",
        va="center",
        fontsize=12,
        bbox=box_style_out,
    )

    ax.annotate("", xy=(0.36, 0.55), xytext=(0.27, 0.55), arrowprops=dict(arrowstyle="->", linewidth=1.8))
    ax.annotate("", xy=(0.71, 0.55), xytext=(0.62, 0.55), arrowprops=dict(arrowstyle="->", linewidth=1.8))

    fig.suptitle("Window feature extraction scheme", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(WINDOW_FEATURE_SCHEME_PNG, dpi=220)
    plt.close(fig)


def run_pipeline(window_size_sec: float = 10.0, step_size_sec: float = 2.0) -> Dict[str, object]:
    ensure_directories()
    prep_summary = run_preparation(window_size_sec=window_size_sec, step_size_sec=step_size_sec)
    metrics_summary = train_random_forest(random_state=42, test_size=0.2)

    build_window_feature_scheme(
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        feature_count=int(prep_summary["feature_count"]),
    )

    return {
        "preparation": prep_summary,
        "metrics": metrics_summary,
        "window_feature_scheme_png": str(WINDOW_FEATURE_SCHEME_PNG),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full data preparation and training pipeline.")
    parser.add_argument("--window-size-sec", type=float, default=10.0, help="Sliding window size, s.")
    parser.add_argument("--step-size-sec", type=float, default=2.0, help="Sliding window step, s.")
    args = parser.parse_args()

    summary = run_pipeline(window_size_sec=args.window_size_sec, step_size_sec=args.step_size_sec)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
