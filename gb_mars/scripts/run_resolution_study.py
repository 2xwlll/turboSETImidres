#!/usr/bin/env python3
"""Run turboSETI for multiple resolution cases and collect a compact summary.

Example:
    python scripts/run_resolution_study.py \
        --case tutorial=/path/to/tutorial_file.h5 \
        --case midpoint=/path/to/midpoint_file.h5 \
        --case midres=/path/to/midres_file.h5 \
        --output-dir /datax/scratch/wlll2x/results/resolution_study \
        --max-drift 4 --snr 10

If you need a true midpoint resolution but only have tutorial and MidRes files,
create a rebinned copy with a frequency step halfway between the two and use
that file as the midpoint case.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run turboSETI across multiple resolution cases")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        required=True,
        help="Case specification as label=/path/to/file.h5. Repeat for each case.",
    )
    parser.add_argument(
        "--output-dir",
        default="/datax/scratch/wlll2x/results/resolution_study",
        help="Directory where per-case outputs and summaries will be written",
    )
    parser.add_argument("--max-drift", type=float, default=4.0)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--f-start", type=float, default=None)
    parser.add_argument("--f-stop", type=float, default=None)
    return parser.parse_args()


def parse_case(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Case specification must be label=/path/to/file.h5, got: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError("Case label cannot be empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return label, path


def load_waterfall(path: str, f_start: Optional[float] = None, f_stop: Optional[float] = None) -> Waterfall:
    kwargs = {"load_data": True}
    if f_start is not None:
        kwargs["f_start"] = f_start
    if f_stop is not None:
        kwargs["f_stop"] = f_stop
    return Waterfall(path, **kwargs)


def infer_resolution(header: dict) -> Optional[float]:
    for key in ("foff", "DELTAF", "deltaf"):
        if key in header:
            value = float(header[key])
            return abs(value)
    return None


def save_waterfall_plot(data: np.ndarray, out_path: str, title: str) -> None:
    plot_data = np.log10(data + 1)
    vmin = np.percentile(plot_data, 5)
    vmax = np.percentile(plot_data, 95)

    plt.figure(figsize=(12, 6))
    plt.imshow(plot_data, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.xlabel("Frequency Channel")
    plt.ylabel("Time Integration")
    plt.title(title)
    plt.colorbar(label="log10(power)")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def run_case(label: str, path: str, output_dir: Path, max_drift: float, snr: float, f_start: Optional[float], f_stop: Optional[float]) -> dict:
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running case: {label} ===")
    print(path)

    t0 = time.time()
    wf = load_waterfall(path, f_start=f_start, f_stop=f_stop)
    runtime_load = time.time() - t0

    header = wf.header
    data = wf.data[:, 0, :]
    resolution = infer_resolution(header)

    waterfall_plot = case_dir / f"{label}_waterfall.png"
    save_waterfall_plot(data, str(waterfall_plot), f"{label}: waterfall")

    print("Data shape:", data.shape)
    print("Resolution:", resolution)

    t_search = time.time()
    fd = FindDoppler(
        datafile=path,
        out_dir=str(case_dir),
        min_drift=-max_drift,
        max_drift=max_drift,
        snr=snr,
    )
    fd.search()
    runtime_search = time.time() - t_search

    output_files = sorted([p.name for p in case_dir.iterdir() if p.is_file()])
    candidate_files = [name for name in output_files if name.endswith((".dat", ".csv", ".txt"))]

    summary = {
        "label": label,
        "input_file": path,
        "output_dir": str(case_dir),
        "shape": list(data.shape),
        "resolution": resolution,
        "max_drift": max_drift,
        "snr": snr,
        "runtime_load_sec": round(runtime_load, 3),
        "runtime_search_sec": round(runtime_search, 3),
        "runtime_total_sec": round(runtime_load + runtime_search, 3),
        "candidate_output_files": len(candidate_files),
        "output_files": output_files,
    }

    with open(case_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with open(output_dir / "summary.csv", "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        if fh.tell() == 0:
            writer.writeheader()
        writer.writerow(summary)

    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CSV header
    csv_path = output_dir / "summary.csv"
    if csv_path.exists():
        csv_path.unlink()

    results = []
    for spec in args.cases:
        label, path = parse_case(spec)
        result = run_case(
            label=label,
            path=path,
            output_dir=output_dir,
            max_drift=args.max_drift,
            snr=args.snr,
            f_start=args.f_start,
            f_stop=args.f_stop,
        )
        results.append(result)

    print("\nCompleted runs:")
    for item in results:
        print(f"- {item['label']}: {item['runtime_total_sec']} s, resolution={item['resolution']}, candidate_files={item['candidate_output_files']}")


if __name__ == "__main__":
    main()
