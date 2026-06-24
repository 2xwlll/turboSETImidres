from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Voyager file (KNOWN GOOD)
# --------------------------------------------------

voyager_file = (
    "/datag/public/voyager_2020/"
    "spliced_blc00010203040506o7o0111213141516o7o021222324252627_"
    "guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"
)

print("File exists:", os.path.exists(voyager_file))
print(voyager_file)

# --------------------------------------------------
# LEVEL 1 WINDOW
# --------------------------------------------------

f_start = 11100.0
f_stop  = 11110.0

print("Loading Level 1 slice...")

wf = Waterfall(
    voyager_file,
    f_start=f_start,
    f_stop=f_stop,
    load_data=True
)

print("Data shape:", wf.data.shape)

# --------------------------------------------------
# SAVE LEVEL 1 HDF5
# --------------------------------------------------

level1_file = "voyager_level1.h5"

print("Writing Level 1 file...")
wf.write_to_hdf5(level1_file)

print("Saved:", level1_file)

# --------------------------------------------------
# SAVE WATERFALL IMAGE
# --------------------------------------------------

plt.figure(figsize=(10,5))

plt.imshow(
    wf.data[:,0,:],
    aspect='auto',
    origin='lower'
)

plt.xlabel("Frequency Bin")
plt.ylabel("Time Integration")
plt.title("Level 1 Voyager Waterfall")

plt.colorbar(label="Power")

plt.savefig("level1_waterfall.png", dpi=300)
plt.close()

print("Saved level1_waterfall.png")

# --------------------------------------------------
# TURBOSETI ON LEVEL 1 FILE
# --------------------------------------------------

fd = FindDoppler(
    level1_file,
    max_drift=4,
    snr=10
)

fd.search()

print("Level 1 complete")