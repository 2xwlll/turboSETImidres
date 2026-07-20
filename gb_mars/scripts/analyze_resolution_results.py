#!/usr/bin/env python3
"""Summarize the results produced by run_resolution_study.py."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    summary_csv = Path("/datax/scratch/wlll2x/results/resolution_study/summary.csv")
    if not summary_csv.exists():
        raise FileNotFoundError(f"No summary file found at {summary_csv}")

    df = pd.read_csv(summary_csv)
    print(df[["label", "resolution", "runtime_total_sec", "candidate_output_files", "shape"]].to_string(index=False))

    if len(df) >= 2:
        print("\nRelative comparison:")
        base = df.iloc[0]
        for _, row in df.iterrows():
            if row["label"] == base["label"]:
                continue
            print(
                f"{row['label']} vs {base['label']}: "
                f"runtime x{row['runtime_total_sec'] / base['runtime_total_sec']:.2f}, "
                f"candidate files x{row['candidate_output_files'] / base['candidate_output_files']:.2f}"
            )


if __name__ == "__main__":
    main()
