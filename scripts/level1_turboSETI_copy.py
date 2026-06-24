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

SNR_THRESHOLD = 10
MAX_DRIFT = 4

files = sorted(glob.glob(BASE_DIR + "*VOYAGER-1_00*.h5"))

print(f"Found {len(files)} files")

# --------------------------------------------------
# STORAGE
# --------------------------------------------------

results = []

# --------------------------------------------------
# PROCESS LOOP
# --------------------------------------------------

for i, f in enumerate(files):

    print("\n==============================")
    print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")

    try:
        # -----------------------------
        # Load header ONLY (fast)
        # -----------------------------
        wf = Waterfall(f, load_data=False)

        fch1 = float(wf.header["fch1"])
        foff = float(wf.header["foff"])

        # -----------------------------
        # Run TurboSETI
        # -----------------------------
        t0 = time.time()

        fd = FindDoppler(
            f,
            max_drift=MAX_DRIFT,
            snr=SNR_THRESHOLD
        )

        fd.search()

        runtime = time.time() - t0

        # -----------------------------
        # SAFE RESULT EXTRACTION
        # -----------------------------
        # TurboSETI stores results in-memory (this is the key fix)
        hits = getattr(fd, "hits", None)

        if hits is None or len(hits) == 0:
            print("No hits")
            continue

        hits = np.array(hits)

        # Expected format:
        # [SNR, drift_rate, index, ...]
        if hits.ndim != 2 or hits.shape[1] < 3:
            print("Unexpected hit format, skipping")
            continue

        # Pick strongest detection
        best = hits[np.argmax(hits[:, 0])]

        snr = float(best[0])
        drift = float(best[1])
        index = int(best[2])

        # -----------------------------
        # Convert index → frequency
        # -----------------------------
        freq_mhz = fch1 + index * foff

        print(f"SNR={snr:.2f} | drift={drift:.4f} Hz/s | freq={freq_mhz:.6f} MHz")

        # -----------------------------
        # STORE RESULT
        # -----------------------------
        results.append({
            "file": os.path.basename(f),
            "snr": snr,
            "drift": drift,
            "freq_mhz": freq_mhz,
            "index": index,
            "runtime": runtime
        })

    except Exception as e:
        print("ERROR:", e)
        continue

# --------------------------------------------------
# BUILD DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(results)

print("\n================ RESULTS ================\n")
print(df)

df.to_csv(os.path.join(OUTDIR, "voyager_clean_results.csv"), index=False)

# --------------------------------------------------
# DIAGNOSTIC PLOTS
# --------------------------------------------------

if len(df) > 0:

    # Frequency stability
    plt.figure()
    plt.plot(df["freq_mhz"], marker="o")
    plt.title("Frequency vs Scan")
    plt.xlabel("Scan index")
    plt.ylabel("MHz")
    plt.savefig(os.path.join(OUTDIR, "freq_stability.png"))
    plt.close()

    # Drift stability
    plt.figure()
    plt.plot(df["drift"], marker="o")
    plt.title("Drift vs Scan")
    plt.xlabel("Scan index")
    plt.ylabel("Hz/s")
    plt.savefig(os.path.join(OUTDIR, "drift_stability.png"))
    plt.close()

    # SNR stability
    plt.figure()
    plt.plot(df["snr"], marker="o")
    plt.title("SNR vs Scan")
    plt.xlabel("Scan index")
    plt.ylabel("SNR")
    plt.savefig(os.path.join(OUTDIR, "snr_stability.png"))
    plt.close()

print("\nDone. Results saved to:", OUTDIR)