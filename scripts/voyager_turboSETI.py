#!/usr/bin/env python3
"""Run turboSETI across raw-data resolution cases and summarize detection performance.

This script analyzes one or more input files or directories and writes:
- per-case candidate summaries
- waterfall plots
- morphology feature tables and clusters
- anomaly candidates
- overall summary CSV
- ON/OFF event comparisons when multiple DAT files are available
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event import find_events, read_dat


DEFAULT_OUTDIR = "/datax/scratch/wlll2x/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run turboSETI and analyze performance, bandwidth scaling, morphology, and ON/OFF comparisons"
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        required=True,
        help="Case specification as label=/path/to/file-or-directory. Repeat for each case.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTDIR, help="Base output directory")
    parser.add_argument("--max-drift", type=float, default=4.0, help="Maximum drift rate in Hz/s")
    parser.add_argument("--min-drift", type=float, default=0.0, help="Minimum drift rate in Hz/s")
    parser.add_argument("--snr", type=float, default=10.0, help="Minimum SNR threshold")
    parser.add_argument("--expected-freq", type=float, default=None, help="Expected narrowband center frequency in MHz for midres region analysis")
    parser.add_argument("--expected-freq-window", type=float, default=0.01, help="Half-width of expected frequency region in MHz for midres analysis")
    parser.add_argument("--stationary-zscore-threshold", type=float, default=4.0, help="Z-score threshold for identifying persistent narrowband excess in the expected region")
    parser.add_argument("--f-start", type=float, default=None, help="Start frequency (MHz) to trim input data")
    parser.add_argument("--f-stop", type=float, default=None, help="Stop frequency (MHz) to trim input data")
    parser.add_argument("--cluster-k", type=int, default=3, help="Number of morphology clusters")
    parser.add_argument(
        "--on-off-first",
        choices=("ON", "OFF"),
        default="ON",
        help="Assume the cadence starts with ON or OFF when comparing an ON/OFF sequence",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold used to flag candidate anomalies",
    )
    return parser.parse_args()


def parse_case(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Case specification must be label=/path/to/file-or-directory, got: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError("Case label cannot be empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return label, path


def resolve_case_specs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    return [parse_case(spec) for spec in args.cases]


def discover_inputs(path: str) -> List[str]:
    if os.path.isdir(path):
        matches: List[str] = []

        # ONLY mid-res Voyager products
        matches.extend(glob.glob(os.path.join(path, "*.rawspec.0002.h5")))

        # optional: safety filter (in case nested dirs appear later)
        filtered: List[str] = []
        for fname in matches:
            if "rawspec.0002.h5" in fname:
                filtered.append(fname)

        return sorted(set(filtered))

    return [path]


def infer_resolution(header: dict) -> Optional[float]:
    for key in ("foff", "DELTAF", "deltaf"):
        if key in header and header[key] is not None:
            try:
                return abs(float(header[key]))
            except (TypeError, ValueError):
                continue
    return None


def infer_bandwidth(header: dict, data_shape: Sequence[int]) -> float:
    resolution = infer_resolution(header)
    if resolution is not None and len(data_shape) >= 2:
        return resolution * float(data_shape[1])

    if len(data_shape) >= 2:
        foff = float(header.get("foff", 0.0))
        return abs(foff) * float(data_shape[1])
    return 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_frequency_axis(header: dict, nchan: int) -> np.ndarray:
    fch1 = safe_float(header.get("fch1", header.get("FCH1", header.get("f_start", 0.0))))
    foff = safe_float(header.get("foff", header.get("DELTAF", header.get("deltaf", 0.0))))
    f_start = safe_float(header.get("f_start", 0.0))
    f_stop = safe_float(header.get("f_stop", 0.0))

    if foff == 0.0 and nchan > 1 and f_stop != f_start:
        foff = (f_stop - f_start) / max(nchan - 1, 1)
    if foff == 0.0:
        foff = 1.0

    return fch1 + np.arange(nchan, dtype=float) * foff


def robust_spectrum_stats(spectrum: np.ndarray) -> Tuple[float, float, np.ndarray]:
    median = float(np.median(spectrum))
    mad = float(np.median(np.abs(spectrum - median)))
    sigma = max(1.4826 * mad, 1e-9)
    zscore = (spectrum - median) / sigma
    return median, sigma, zscore


def stationary_line_report(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    expected_freq: Optional[float],
    window_mhz: float,
    zscore_threshold: float,
) -> Dict[str, Any]:
    median, sigma, zscore = robust_spectrum_stats(spectrum)
    report = {
        "spectrum_median_power": median,
        "spectrum_sigma_power": sigma,
        "spectrum_max_zscore": float(np.max(zscore)) if zscore.size else 0.0,
        "spectrum_threshold_excess_bins": int(np.sum(zscore >= zscore_threshold)),
    }
    if expected_freq is not None:
        region_mask = np.abs(freqs_mhz - expected_freq) <= window_mhz
        region_spectrum = spectrum[region_mask]
        region_zscore = zscore[region_mask]
        report.update(
            {
                "expected_freq_mhz": expected_freq,
                "expected_freq_window_mhz": window_mhz,
                "expected_region_bin_count": int(np.sum(region_mask)),
                "expected_region_mean_power": float(np.mean(region_spectrum)) if region_spectrum.size else float("nan"),
                "expected_region_max_zscore": float(np.max(region_zscore)) if region_zscore.size else float("nan"),
                "expected_region_excess_bins": int(np.sum(region_zscore >= zscore_threshold)),
                "expected_region_excess_fraction": float(np.mean(region_zscore >= zscore_threshold)) if region_zscore.size else 0.0,
            }
        )
    return report


def count_region_candidates(df: pd.DataFrame, expected_freq: Optional[float], window_mhz: float) -> Dict[str, Any]:
    if df.empty or expected_freq is None:
        return {
            "region_candidate_count": 0,
            "region_candidate_mean_snr": 0.0,
            "region_candidate_max_snr": 0.0,
        }
    mask = np.abs(df["FrequencyCentreMHz"] - expected_freq) <= window_mhz
    region_df = df[mask]
    return {
        "region_candidate_count": int(len(region_df)),
        "region_candidate_mean_snr": float(region_df["SNR_abs"].mean()) if len(region_df) else 0.0,
        "region_candidate_max_snr": float(region_df["SNR_abs"].max()) if len(region_df) else 0.0,
    }


def save_average_spectrum(freqs_mhz: np.ndarray, spectrum: np.ndarray, out_path: str) -> None:
    df = pd.DataFrame({"frequency_mhz": freqs_mhz, "avg_power": spectrum})
    df.to_csv(out_path, index=False)


def source_scan_type(source_name: str) -> str:
    source_lower = source_name.lower()
    if "off" in source_lower:
        return "OFF"
    if "on" in source_lower:
        return "ON"
    return "ON"


def save_waterfall_plot(data: np.ndarray, out_path: str, title: str) -> None:
    if data.ndim != 2:
        data = np.asarray(data).reshape(-1, 1)

    plot_data = np.log10(np.asarray(data, dtype=float) + 1)
    if plot_data.size == 0:
        return

    mean_spec = np.mean(plot_data, axis=0)
    peak_idx = int(np.argmax(mean_spec))
    nchan = plot_data.shape[1]
    window = max(200, min(5000, int(0.01 * nchan)))
    left = max(0, peak_idx - window // 2)
    right = min(nchan, left + window)
    left = max(0, right - window)
    zoom_data = plot_data[:, left:right]

    vmin = np.percentile(zoom_data, 5)
    vmax = np.percentile(zoom_data, 95)

    plt.figure(figsize=(12, 6))
    plt.imshow(zoom_data, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.xlabel("Frequency Channel")
    plt.ylabel("Time Integration")
    plt.title(f"{title} (zoomed around strongest spectral feature)")
    plt.colorbar(label="log10(power)")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def safe_read_dat(dat_path: Path) -> pd.DataFrame:
    if not dat_path.exists() or dat_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return read_dat(str(dat_path))
    except Exception as exc:
        warnings.warn(f"Unable to parse DAT file {dat_path}: {exc}")
        return pd.DataFrame()


def candidate_feature_table(df: pd.DataFrame, header: dict, data_shape: Sequence[int]) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["DriftAbs"] = df["DriftRate"].abs()
    df["SNR_abs"] = df["SNR"].abs()
    df["WidthMHz"] = (df["FreqEnd"] - df["FreqStart"]).abs()
    df["WidthHz"] = df["WidthMHz"] * 1e6
    df["FrequencyCentreMHz"] = df[["FreqStart", "FreqEnd"]].mean(axis=1)
    bandwidth_mhz = infer_bandwidth(header, data_shape)
    fch1 = float(header.get("fch1", 0.0))
    df["FreqRel"] = (df["Freq"] - fch1) / max(bandwidth_mhz, 1e-9)
    df["CandidateDensityPerMHz"] = len(df) / max(bandwidth_mhz, 1e-9)
    return df


def kmeans_numpy(X: np.ndarray, n_clusters: int, max_iters: int = 50, tol: float = 1e-6) -> np.ndarray:
    if len(X) == 0 or n_clusters <= 0:
        return np.array([], dtype=int)

    n_samples = X.shape[0]
    n_clusters = min(n_clusters, n_samples)
    rng = np.random.default_rng(0)
    centroids = X[rng.choice(n_samples, size=n_clusters, replace=False)]
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_idx in range(n_clusters):
            members = X[labels == cluster_idx]
            if len(members) > 0:
                centroids[cluster_idx] = members.mean(axis=0)
    return labels


def cluster_morphology(df: pd.DataFrame, cluster_k: int) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)

    feature_cols = ["SNR_abs", "DriftAbs", "WidthHz", "FullNumHitsInRange"]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing candidate columns for clustering: {missing}")

    X = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    labels = kmeans_numpy(X, cluster_k)
    return pd.Series(labels, index=df.index, name="morphology_cluster")


def summarize_candidates(df: pd.DataFrame, bandwidth_mhz: float) -> Dict[str, Any]:
    if df.empty:
        return {
            "candidate_count": 0,
            "candidate_density_per_mhz": 0.0,
            "mean_snr": 0.0,
            "median_width_hz": 0.0,
            "mean_abs_drift_rate": 0.0,
        }

    return {
        "candidate_count": int(len(df)),
        "candidate_density_per_mhz": float(len(df) / max(bandwidth_mhz, 1e-9)),
        "mean_snr": float(df["SNR_abs"].mean()),
        "median_width_hz": float(df["WidthHz"].median()),
        "mean_abs_drift_rate": float(df["DriftAbs"].mean()),
    }


def detect_voyager_hit(input_path: str, source_name: str, candidate_count: int) -> bool:
    return (
        candidate_count > 0
        and ("voyager" in source_name.lower() or "voyager" in Path(input_path).name.lower())
    )


def find_on_off_events(dat_paths: Sequence[str], on_off_first: str, args: argparse.Namespace) -> Optional[pd.DataFrame]:
    if len(dat_paths) < 2:
        return None
    return find_events(
        list(dat_paths),
        filter_threshold=3,
        on_off_first=on_off_first,
        SNR_cut=args.snr,
        min_drift_rate=args.min_drift,
        max_drift_rate=args.max_drift,
    )


def find_anomalies(df: pd.DataFrame, percentile: float) -> pd.DataFrame:
    if df.empty:
        return df

    thresholds = {
        "SNR_abs": np.percentile(df["SNR_abs"], percentile),
        "WidthHz": np.percentile(df["WidthHz"], percentile),
        "DriftAbs": np.percentile(df["DriftAbs"], percentile),
    }

    anomalies = df[
        (df["SNR_abs"] > thresholds["SNR_abs"])
        | (df["WidthHz"] > thresholds["WidthHz"])
        | (df["DriftAbs"] > thresholds["DriftAbs"])
    ].copy()
    if anomalies.empty:
        return anomalies

    anomalies["anomaly_score"] = (
        anomalies["SNR_abs"] / max(thresholds["SNR_abs"], 1e-9)
        + anomalies["WidthHz"] / max(thresholds["WidthHz"], 1e-9)
        + anomalies["DriftAbs"] / max(thresholds["DriftAbs"], 1e-9)
    )
    return anomalies.sort_values("anomaly_score", ascending=False)


def build_case_summary(
    input_path: str,
    label: str,
    header: dict,
    source_name: str,
    scan_type: str,
    runtime_load: float,
    runtime_search: float,
    data_shape: Sequence[int],
    candidate_summary: Dict[str, Any],
    dat_path: Path,
    voyager_detected: bool,
    clusters: pd.Series,
    static_metrics: Optional[Dict[str, Any]] = None,
    region_metrics: Optional[Dict[str, Any]] = None,
    status: str = "OK",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    bandwidth_mhz = infer_bandwidth(header, data_shape)
    summary = {
        "label": label,
        "input_file": input_path,
        "scan_type": scan_type,
        "shape": list(data_shape),
        "bandwidth_mhz": float(bandwidth_mhz),
        "runtime_load_sec": round(runtime_load, 3),
        "runtime_search_sec": round(runtime_search, 3),
        "runtime_total_sec": round(runtime_load + runtime_search, 3),
        "candidate_count": candidate_summary["candidate_count"],
        "candidate_density_per_mhz": candidate_summary["candidate_density_per_mhz"],
        "mean_snr": candidate_summary["mean_snr"],
        "median_width_hz": candidate_summary["median_width_hz"],
        "mean_abs_drift_rate": candidate_summary["mean_abs_drift_rate"],
        "voyager_detected": voyager_detected,
        "dat_file": str(dat_path),
        "cluster_count": int(clusters.nunique()) if len(clusters) > 0 else 0,
        "status": status,
        "error": error,
    }
    if static_metrics:
        summary.update(static_metrics)
    if region_metrics:
        summary.update(region_metrics)
    return summary


def process_file(
    input_path: str,
    label: str,
    output_dir: Path,
    max_drift: float,
    min_drift: float,
    snr: float,
    expected_freq: Optional[float],
    expected_freq_window: float,
    stationary_zscore_threshold: float,
    f_start: Optional[float],
    f_stop: Optional[float],
    cluster_k: int,
    anomaly_percentile: float,
) -> dict:
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem
    print(f"\n=== Processing {label}: {input_path} ===")

    load_kwargs = {"load_data": True}
    if f_start is not None:
        load_kwargs["f_start"] = f_start
    if f_stop is not None:
        load_kwargs["f_stop"] = f_stop

    try:
        t0 = time.time()
        wf = Waterfall(input_path, **load_kwargs)
        runtime_load = time.time() - t0

        data = wf.data[:, 0, :]
        header = wf.header
        source_name = str(header.get("source_name", "unknown"))
        scan_type = source_scan_type(source_name)

        freqs_mhz = infer_frequency_axis(header, data.shape[1])
        avg_spectrum = np.mean(data, axis=0)
        static_metrics = stationary_line_report(
            freqs_mhz,
            avg_spectrum,
            expected_freq=expected_freq,
            window_mhz=expected_freq_window,
            zscore_threshold=stationary_zscore_threshold,
        )
        if expected_freq is not None:
            save_average_spectrum(freqs_mhz, avg_spectrum, str(case_dir / f"{stem}_avg_spectrum.csv"))

        waterfall_path = case_dir / f"{stem}_waterfall.png"
        save_waterfall_plot(data, str(waterfall_path), f"{label}: {Path(input_path).name}")

        print("Running turboSETI...")
        t_search = time.time()
        fd = FindDoppler(
            input_path,
            max_drift=max_drift,
            min_drift=min_drift,
            snr=snr,
            out_dir=str(case_dir),
        )
        fd.search()
        runtime_search = time.time() - t_search

        dat_path = case_dir / f"{stem}.dat"
        hits_df = safe_read_dat(dat_path)
        if not hits_df.empty:
            hits_df = candidate_feature_table(hits_df, header, data.shape)
            morphology_labels = cluster_morphology(hits_df, cluster_k)
            hits_df["morphology_cluster"] = morphology_labels
            hits_df.to_csv(case_dir / f"{stem}_candidate_features.csv", index=False)
            anomalies_df = find_anomalies(hits_df, anomaly_percentile)
            anomalies_df.to_csv(case_dir / f"{stem}_anomalies.csv", index=False)
        else:
            morphology_labels = pd.Series(dtype=int)

        region_metrics = count_region_candidates(hits_df, expected_freq, expected_freq_window)
        summary = build_case_summary(
            input_path=input_path,
            label=label,
            header=header,
            source_name=source_name,
            scan_type=scan_type,
            runtime_load=runtime_load,
            runtime_search=runtime_search,
            data_shape=data.shape,
            candidate_summary=summarize_candidates(hits_df, infer_bandwidth(header, data.shape)),
            dat_path=dat_path,
            voyager_detected=detect_voyager_hit(input_path, source_name, len(hits_df)),
            clusters=morphology_labels,
            static_metrics=static_metrics,
            region_metrics=region_metrics,
        )
        with open(case_dir / f"{stem}_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return summary
    except Exception as exc:
        print(f"Failed on {input_path}: {exc}")
        summary = build_case_summary(
            input_path=input_path,
            label=label,
            header={},
            source_name="unknown",
            scan_type="UNKNOWN",
            runtime_load=0.0,
            runtime_search=0.0,
            data_shape=[],
            candidate_summary={
                "candidate_count": 0,
                "candidate_density_per_mhz": 0.0,
                "mean_snr": 0.0,
                "median_width_hz": 0.0,
                "mean_abs_drift_rate": 0.0,
            },
            dat_path=case_dir / f"{stem}.dat",
            voyager_detected=False,
            clusters=pd.Series(dtype=int),
            status="FAILED",
            error=str(exc),
        )
        with open(case_dir / f"{stem}_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return summary


def save_overall_summary(summaries: List[Dict[str, Any]], output_dir: Path) -> None:
    if not summaries:
        return
    summary_csv = output_dir / "summary.csv"
    fieldnames = sorted(summaries[0].keys())
    with open(summary_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    print(f"\nWrote summary CSV to {summary_csv}")


def save_event_results(events: pd.DataFrame, output_dir: Path) -> None:
    if events is None or events.empty:
        print("No ON/OFF events were identified.")
        return
    out_file = output_dir / "on_off_events.csv"
    events.to_csv(out_file, index=False)
    print(f"Wrote ON/OFF event table to {out_file}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_specs = resolve_case_specs(args)
    summaries: List[Dict[str, Any]] = []
    dat_paths: List[str] = []

    for label, path in case_specs:
        inputs = discover_inputs(path)
        if not inputs:
            print(f"No input files found for {label}: {path}")
            continue

        print(f"\nRunning case '{label}' with {len(inputs)} input file(s)")
        for input_path in inputs:
            result = process_file(
                input_path=input_path,
                label=label,
                output_dir=output_dir,
                max_drift=args.max_drift,
                min_drift=args.min_drift,
                snr=args.snr,
                expected_freq=args.expected_freq,
                expected_freq_window=args.expected_freq_window,
                stationary_zscore_threshold=args.stationary_zscore_threshold,
                f_start=args.f_start,
                f_stop=args.f_stop,
                cluster_k=args.cluster_k,
                anomaly_percentile=args.anomaly_percentile,
            )
            summaries.append(result)
            if result.get("dat_file"):
                dat_paths.append(result["dat_file"])

    save_overall_summary(summaries, output_dir)

    if len(dat_paths) >= 2:
        on_off_events = find_on_off_events(dat_paths, args.on_off_first, args)
        if isinstance(on_off_events, pd.DataFrame):
            save_event_results(on_off_events, output_dir)

    print("\nRun complete.")


if __name__ == "__main__":
    main()
