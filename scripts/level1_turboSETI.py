from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------

OUTDIR = "/datax/scratch/wlll2x/results"

os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# LEVEL 1 DATA
# Single coarse channel
# --------------------------------------------------

voyager_file = (
    "/datag/public/voyager_2020/single_coarse_channel/"
    "single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"
)

print("File exists:", os.path.exists(voyager_file))
print(voyager_file)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading single coarse channel...")

wf = Waterfall(voyager_file, load_data=True)

print("Data shape:", wf.data.shape)

print("\nHeader Information")
print("------------------")

for k, v in wf.header.items():
    print(f"{k} = {v}")

# --------------------------------------------------
# WATERFALL PLOT
# --------------------------------------------------

print("\nCreating waterfall plot...")

data = wf.data[:, 0, :]

# log scaling usually looks much better
plot_data = np.log10(data + 1)

plt.figure(figsize=(12, 6))

plt.imshow(
    plot_data,
    aspect="auto",
    origin="lower"
)

plt.xlabel("Frequency Channel")
plt.ylabel("Time Integration")
plt.title("Voyager Level 1 (Single Coarse Channel)")

plt.colorbar(label="log10(Power)")

png_file = os.path.join(
    OUTDIR,
    "level1_waterfall.png"
)

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", png_file)

# --------------------------------------------------
# TURBOSETI
# --------------------------------------------------

print("\nStarting turboSETI...")

t0 = time.time()

fd = FindDoppler(
    voyager_file,
    max_drift=4,
    snr=10
)

fd.search()

runtime = time.time() - t0

print(f"\nTurboSETI runtime = {runtime:.1f} sec")
print("Level 1 complete")