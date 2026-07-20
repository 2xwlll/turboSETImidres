from blimpy import Waterfall
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# File
# --------------------------------------------------

filename = "/datag/public/A_Sign_in_Space_raw_h5/blc33_guppi_60067_72149_EXOMARS16TGO_DIAG_0010.rawspec.0002.h5"

wf = Waterfall(filename)

data = wf.data[:, 0, :]

# --------------------------------------------------
# Choose coarse channel
# --------------------------------------------------

coarse = 9
nfine = 1024

start = coarse * nfine
stop = start + nfine

chunk = data[:, start:stop]

# --------------------------------------------------
# Frequency axis
# --------------------------------------------------

foff = wf.header["foff"]      # MHz/channel
fch1 = wf.header["fch1"]      # MHz

freqs = fch1 + np.arange(data.shape[1]) * foff

# --------------------------------------------------
# Find brightest channel every integration
# --------------------------------------------------

peak_channels = np.argmax(chunk, axis=1) + start
peak_freqs = freqs[peak_channels]

times = np.arange(len(peak_freqs)) * wf.header["tsamp"]

# --------------------------------------------------
# Linear fit
# --------------------------------------------------

coeff = np.polyfit(times, peak_freqs, 1)

slope_mhz = coeff[0]
intercept = coeff[1]

drift_hz = slope_mhz * 1e6

print()
print("===================================")
print(f"Measured drift = {drift_hz:.2f} Hz/s")
print("===================================")
print()

# --------------------------------------------------
# Plot frequency vs time
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(times, peak_freqs,
            s=8,
            label="Measured peak")

plt.plot(
    times,
    np.polyval(coeff, times),
    color="red",
    linewidth=2,
    label=f"Fit ({drift_hz:.2f} Hz/s)"
)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
plt.title("Measured ExoMars Drift")
plt.legend()

plt.tight_layout()
plt.savefig("exomars_drift_fit.png", dpi=200)

# --------------------------------------------------
# Waterfall with fit
# --------------------------------------------------

plt.figure(figsize=(10,7))

plt.imshow(
    chunk,
    aspect="auto",
    origin="lower",
    interpolation="nearest"
)

# convert fitted frequencies back to fine channels

fit_channels = (
    (np.polyval(coeff, times) - fch1) / foff
) - start

plt.plot(
    fit_channels,
    np.arange(len(times)),
    color="red",
    linewidth=2
)

plt.xlabel("Fine channel")
plt.ylabel("Time integration")
plt.title("Waterfall with Best-Fit Drift")

plt.tight_layout()
plt.savefig("exomars_waterfall_fit.png", dpi=200)

print("Saved:")
print("  exomars_drift_fit.png")
print("  exomars_waterfall_fit.png")