from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob
import os
import time

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = "/datag/public/voyager_2020/single_coarse_channel/"
OUTDIR = "/datax/scratch/wlll2x/results"

os.makedirs(OUTDIR, exist_ok=True)

MAX_DRIFT = 4
SNR_THRESHOLD = 10

# --------------------------------------------------
# FIND ONLY VALID SCIENCE FILES
# --------------------------------------------------

all_files = sorted(glob.glob(os.path.join(BASE_DIR, "*.h5")))

files = []

for f in all_files:

    if ".8.0001." in f:
        continue

    if ".0002." in f:
        continue

    if ".0000." not in f:
        continue

    files.append(f)

print(f"\nFound {len(files)} valid .0000 science files")

# --------------------------------------------------
# RESULTS STORAGE
# --------------------------------------------------

results = []

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

for i, filename in enumerate(files):

    print("\n===================================")
    print(f"[{i+1}/{len(files)}]")
    print(os.path.basename(filename))

    try:

        # ------------------------------------------
        # LOAD DATA
        # ------------------------------------------

        wf = Waterfall(filename, load_data=True)

        header = wf.header

        source_name = str(header["source_name"])
        fch1 = float(header["fch1"])
        foff = float(header["foff"])

        print("Source:", source_name)

        if "OFF" in source_name:
            scan_type = "OFF"
        else:
            scan_type = "ON"

        data = wf.data[:, 0, :]

        print("Shape:", data.shape)

        # ------------------------------------------
        # FULL WATERFALL
        # ------------------------------------------

        plot_data = np.log10(data + 1)

        vmin = np.percentile(plot_data, 5)
        vmax = np.percentile(plot_data, 95)

        plt.figure(figsize=(12, 6))

        plt.imshow(
            plot_data,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )

        plt.xlabel("Frequency Channel")
        plt.ylabel("Time Integration")

        plt.title(
            f"{scan_type} : {os.path.basename(filename)}"
        )

        plt.colorbar(label="log10(power)")

        out_png = os.path.join(
            OUTDIR,
            os.path.basename(filename).replace(
                ".h5",
                "_waterfall.png"
            )
        )

        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        plt.close()

        # ------------------------------------------
        # TURBOSETI
        # ------------------------------------------

        print("Running TurboSETI...")

        t0 = time.time()

        fd = FindDoppler(
            filename,
            max_drift=MAX_DRIFT,
            snr=SNR_THRESHOLD
        )

        fd.search()

        runtime = time.time() - t0

        print(f"Runtime = {runtime:.1f} sec")

        # ------------------------------------------
        # MANUAL PEAK SEARCH
        # (independent sanity check)
        # ------------------------------------------

        mean_spec = np.mean(data, axis=0)

        peak_index = np.argmax(mean_spec)

        peak_power = mean_spec[peak_index]

        peak_freq = fch1 + peak_index * foff

        print(
            f"Peak channel = {peak_index}"
        )

        print(
            f"Peak frequency = {peak_freq:.6f} MHz"
        )

        # ------------------------------------------
        # ZOOM AROUND PEAK
        # ------------------------------------------

        window = 5000

        start = max(0, peak_index - window)
        stop = min(data.shape[1], peak_index + window)

        zoom = data[:, start:stop]

        zoom_log = np.log10(zoom + 1)

        zmin = np.percentile(zoom_log, 5)
        zmax = np.percentile(zoom_log, 95)

        plt.figure(figsize=(12, 6))

        plt.imshow(
            zoom_log,
            aspect="auto",
            origin="lower",
            vmin=zmin,
            vmax=zmax
        )

        plt.xlabel("Frequency Channel")
        plt.ylabel("Time Integration")

        plt.title(
            f"Zoom Around Strongest Signal\n"
            f"{peak_freq:.6f} MHz"
        )

        plt.colorbar(label="log10(power)")

        zoom_png = os.path.join(
            OUTDIR,
            os.path.basename(filename).replace(
                ".h5",
                "_zoom.png"
            )
        )

        plt.savefig(
            zoom_png,
            dpi=250,
            bbox_inches="tight"
        )

        plt.close()

        # ------------------------------------------
        # SAVE RESULTS
        # ------------------------------------------

        results.append({

            "file":
                os.path.basename(filename),

            "type":
                scan_type,

            "peak_freq_mhz":
                peak_freq,

            "peak_channel":
                peak_index,

            "peak_power":
                peak_power,

            "runtime_sec":
                runtime

        })

    except Exception as e:

        print("\nERROR:")
        print(e)

# --------------------------------------------------
# SAVE TABLE
# --------------------------------------------------

df = pd.DataFrame(results)

csv_file = os.path.join(
    OUTDIR,
    "voyager_summary.csv"
)

df.to_csv(csv_file, index=False)

print("\n===================================")
print("SUMMARY")
print("===================================")

print(df)

print("\nSaved:")
print(csv_file)

# --------------------------------------------------
# STABILITY PLOTS
# --------------------------------------------------

if len(df) > 0:

    plt.figure()

    plt.plot(
        df["peak_freq_mhz"],
        marker="o"
    )

    plt.title("Peak Frequency Stability")

    plt.xlabel("Scan")
    plt.ylabel("Frequency (MHz)")

    plt.savefig(
        os.path.join(
            OUTDIR,
            "frequency_stability.png"
        )
    )

    plt.close()

    plt.figure()

    plt.plot(
        df["peak_power"],
        marker="o"
    )

    plt.title("Peak Power Stability")

    plt.xlabel("Scan")
    plt.ylabel("Power")

    plt.savefig(
        os.path.join(
            OUTDIR,
            "power_stability.png"
        )
    )

    plt.close()

print("\nDone.")