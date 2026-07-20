import re
import numpy as np
import glob

path = "voyager_midres_snr3_nfpc1024/voyager_midres/*.dat"
files = glob.glob(path)

def extract_drifts(file):
    drifts = []

    with open(file, "r") as f:
        for line in f:
            # match numeric drift rows (turboSETI format)
            if re.match(r"^\d+\s+-?\d+\.\d+", line):
                parts = line.split()
                try:
                    drifts.append(float(parts[1]))
                except:
                    pass

    return np.array(drifts)

all_drifts = []

for f in files:
    d = extract_drifts(f)
    print(f, "drifts found:", len(d))

    if len(d) > 0:
        print("unique sample:", np.unique(np.round(d, 6))[:5])

    all_drifts.extend(d)

all_drifts = np.array(all_drifts)

print("\nGLOBAL UNIQUE DRIFTS:")
print(np.unique(np.round(all_drifts, 6)))