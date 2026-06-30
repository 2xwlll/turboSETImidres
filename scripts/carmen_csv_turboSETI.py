#!/usr/bin/env python3
"""Run turboSETI on HDF5 files listed in the subset CSV and generate waterfall plots.

This script reads the CSV at `scripts/carmenes_1688mhz_subset.csv` by default, loads
each HDF5 file using blimpy, generates a resolution-aware waterfall plot, and runs
FindDoppler to write turboSETI .dat output files for each case.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from blimpy import Waterfall
except ImportError as exc:
    raise ImportError(
        "blimpy is required for this script. Install it with `pip install blimpy` "
        "or activate the environment that contains blimpy."
    ) from exc

try:
    from turbo_seti.find_doppler.find_doppler import FindDoppler
    from turbo_seti.find_event.find_event import read_dat
except ImportError as exc:
    raise ImportError(
        "turbo_seti is required for this script. Install it from the repo or via pip "
        "and make sure it is importable in the current Python environment."
    ) from exc

DEFAULT_CSV = Path(__file__).resolve().parent / "carmenes_1688mhz_subset.csv"
DEFAULT_OUTDIR = Path("/datax/scratch/wlll2x/results/carmen_csv_turboSETI")
MAX_PLOT_POINTS = 2200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run turboSETI for each HDF5 path in a CSV and save waterfall plots."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to the subset CSV containing .h5 path entries",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTDIR),
        help="Base directory where turboSETI outputs and waterfall images will be written",
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=4.0,
        help="Maximum absolute drift rate in Hz/s for turboSETI",
    )
    parser.add_argument(
        "--min-drift",
        type=float,
        default=0.00001,
        help="Minimum drift rate in Hz/s for turboSETI",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=10.0,
        help="Minimum SNR threshold for turboSETI",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of CSV rows to process",
    )
    parser.add_argument(
        "--targets",
        action="append",
        default=[],
        help="Optional target names to filter by (repeatable).",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate rows by .h5 path before processing.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows when the .h5 input file does not exist",
    )
    return parser.parse_args()


def sanitize_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("_.-") or "case"


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def read_subset_csv(csv_path: Path, targets: List[str], dedupe: bool) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    target_filters = [t.strip().lower() for t in targets if t.strip()]
    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row:
                continue
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            h5_path = row.get(".h5 path") or row.get(".h5_path") or row.get("h5 path")
            if not h5_path:
                continue
            row["h5_path"] = h5_path.strip()
            row["dat_path"] = (row.get(".dat path") or row.get(".dat_path") or "").strip()
            if target_filters:
                target_name = str(row.get("Target", "")).strip().lower()
                if target_name not in target_filters:
                    continue
            rows.append(row)

    if dedupe:
        seen: set[str] = set()
        unique_rows: List[Dict[str, str]] = []
        for row in rows:
            path = row["h5_path"]
            if path in seen:
                continue
            seen.add(path)
            unique_rows.append(row)
        rows = unique_rows

    return rows


def normalize_mhz(value: float) -> float:
    if value is None:
        return 0.0
    abs_value = abs(value)
    if abs_value > 1e5:
        return value / 1e6
    return value


def infer_frequency_axis(header: dict, nchan: int) -> Tuple[np.ndarray, float]:
    fch1 = safe_float(header.get("fch1") or header.get("FCH1") or header.get("FREQ"))
    foff = safe_float(header.get("foff") or header.get("DELTAF") or header.get("deltaf"))

    fch1 = normalize_mhz(fch1)
    foff = normalize_mhz(foff)
    if foff == 0.0:
        foff = 1.0

    freqs_mhz = fch1 + np.arange(nchan, dtype=float) * foff
    resolution_hz = abs(foff) * 1e6

    return freqs_mhz, resolution_hz


def rebin_2d(data: np.ndarray, max_points: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("rebin_2d expects a 2D array")

    n_time, n_freq = data.shape
    time_factor = max(1, math.ceil(n_time / max_points))
    freq_factor = max(1, math.ceil(n_freq / max_points))

    if time_factor == 1 and freq_factor == 1:
        return data

    trimmed_time = (n_time // time_factor) * time_factor
    trimmed_freq = (n_freq // freq_factor) * freq_factor
    data = data[:trimmed_time, :trimmed_freq]

    data = data.reshape(trimmed_time // time_factor, time_factor, trimmed_freq // freq_factor, freq_factor)
    return data.mean(axis=(1, 3))


def save_waterfall_plot(
    data: np.ndarray,
    freqs_mhz: np.ndarray,
    tsamp: float,
    output_path: Path,
    title: str,
) -> None:
    if data.ndim == 3:
        data = data[:, 0, :]

    if data.ndim != 2:
        raise ValueError(f"Expected 2D waterfall data, got shape {data.shape}")

    data = np.asarray(data, dtype=float)
    data = data.clip(min=0.0)
    if data.size == 0:
        raise ValueError("Waterfall data is empty")

    original_time, original_freq = data.shape
    data = data + 1.0
    data_db = 10.0 * np.log10(data)

    if original_time > MAX_PLOT_POINTS or original_freq > MAX_PLOT_POINTS:
        data_db = rebin_2d(data_db, MAX_PLOT_POINTS)
        freqs_mhz = np.linspace(freqs_mhz[0], freqs_mhz[-1], data_db.shape[1])

    duration_s = original_time * tsamp
    extent = (float(freqs_mhz[0]), float(freqs_mhz[-1]), 0.0, float(duration_s))

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        data_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power [dB]")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Time [s]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_candidate_dataset(dat_paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dat_path in dat_paths:
        try:
            df = read_dat(dat_path)
        except Exception as exc:
            print(f"Warning: could not read DAT {dat_path}: {exc}")
            continue
        if not df.empty:
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def compute_zoom_bounds(hits_df: pd.DataFrame, pad_mhz: float = 0.1) -> Tuple[float, float]:
    fmin = float(hits_df["FreqStart"].min())
    fmax = float(hits_df["FreqEnd"].max())
    span = max(fmax - fmin, 0.1)
    padding = max(pad_mhz, 0.2 * span)
    return max(0.0, fmin - padding), fmax + padding


def plot_hits_waterfall(
    input_path: Path,
    hits_df: pd.DataFrame,
    output_path: Path,
    label: str,
) -> Tuple[Optional[List[int]], Optional[float]]:
    if hits_df.empty:
        return None, None

    f_start, f_stop = compute_zoom_bounds(hits_df)
    wf = Waterfall(str(input_path), f_start=f_start, f_stop=f_stop, load_data=True)
    data = wf.data
    if data.ndim == 3 and data.shape[1] >= 1:
        data = data[:, 0, :]
    if data.ndim != 2:
        raise ValueError(f"Unexpected Waterfall data shape {data.shape}")

    freqs_mhz, resolution_hz = infer_frequency_axis(wf.header, data.shape[1])
    tsamp = safe_float(wf.header.get("tsamp"), default=1.0)
    title = f"{label} waterfall around candidates"
    save_waterfall_plot(data, freqs_mhz, tsamp, output_path, title)
    return [int(data.shape[0]), int(data.shape[1])], resolution_hz


def process_row(
    row: Dict[str, str],
    output_dir: Path,
    max_drift: float,
    min_drift: float,
    snr: float,
) -> Dict[str, object]:
    target = row.get("Target", "unknown")
    session = row.get("Session", "unknown")
    cadence_id = row.get("Cadence ID", "")
    frequency = row.get("Frequency", "")
    h5_path = row["h5_path"]

    label = sanitize_label(f"{target}_{session}_{cadence_id}")
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(h5_path)
    stem = input_path.stem
    summary: Dict[str, object] = {
        "label": label,
        "target": target,
        "session": session,
        "cadence_id": cadence_id,
        "frequency_mhz": frequency,
        "h5_path": str(input_path),
        "waterfall_png": "",
        "dat_files": [],
        "resolution_hz": None,
        "shape": None,
        "runtime_search_sec": None,
        "error": None,
    }

    if not input_path.exists():
        summary["error"] = f"Input file not found: {input_path}"
        return summary

    try:
        print(f"Running turboSETI on {input_path} (label={label})")
        t_search = time.time()
        fd = FindDoppler(
            datafile=str(input_path),
            max_drift=max_drift,
            min_drift=min_drift,
            snr=snr,
            out_dir=str(case_dir),
        )
        fd.search()
        runtime_search = time.time() - t_search
        summary["runtime_search_sec"] = round(runtime_search, 3)

        dat_files = [str(p) for p in sorted(case_dir.glob("*.dat"))]
        summary["dat_files"] = dat_files

        hits_df = build_candidate_dataset(dat_files)
        if hits_df.empty:
            print(f"No candidates found for {label}; skipping waterfall plot.")
            return summary

        plot_path = case_dir / f"{stem}_candidate_waterfall.png"
        shape, resolution_hz = plot_hits_waterfall(input_path, hits_df, plot_path, label)
        summary["waterfall_png"] = str(plot_path)
        summary["shape"] = shape
        summary["resolution_hz"] = round(resolution_hz, 6) if resolution_hz is not None else None
    except Exception as exc:
        summary["error"] = str(exc)

    return summary


def write_summary_csv(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "target",
        "session",
        "cadence_id",
        "frequency_mhz",
        "h5_path",
        "waterfall_png",
        "dat_files",
        "resolution_hz",
        "shape",
        "runtime_load_sec",
        "runtime_search_sec",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_to_write = row.copy()
            if isinstance(row_to_write.get("dat_files"), list):
                row_to_write["dat_files"] = ";".join(row_to_write["dat_files"])
            if isinstance(row_to_write.get("shape"), list):
                row_to_write["shape"] = "x".join(str(x) for x in row_to_write["shape"])
            writer.writerow(row_to_write)


def main() -> None:
    args = parse_args()
    subset_rows = read_subset_csv(Path(args.csv), args.targets, args.dedupe)
    if args.limit is not None:
        subset_rows = subset_rows[: args.limit]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for row in subset_rows:
        if args.skip_missing:
            h5_path = row.get("h5_path", "")
            if not Path(h5_path).exists():
                print(f"Skipping missing input: {h5_path}")
                continue
        result = process_row(
            row,
            output_dir=output_dir,
            max_drift=args.max_drift,
            min_drift=args.min_drift,
            snr=args.snr,
        )
        results.append(result)

    summary_path = output_dir / "summary.csv"
    write_summary_csv(summary_path, results)
    print(f"Saved summary file: {summary_path}")

    print("Completed processing")
    for result in results:
        print(
            f"{result['label']}: waterfall={result['waterfall_png']} dat_files={len(result.get('dat_files', []))} "
            f"search={result.get('runtime_search_sec')}s"
        )


if __name__ == "__main__":
    main()
