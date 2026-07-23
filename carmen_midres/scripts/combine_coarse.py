from pathlib import Path

# Directory containing cadence_000, cadence_001, cadence_002
INPUT_DIR = Path("/datax/scratch/wlll2x/carmen_midres_bliss")

# Where the merged files will be written
OUTPUT_DIR = Path("/datax/scratch/wlll2x/carmen_midres_bliss_combined")
OUTPUT_DIR.mkdir(exist_ok=True)

#clear em

import shutil

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True)

# Find every coarse-channel BLISS output
files = sorted(INPUT_DIR.rglob("*_coarse_*.dat"))

print(f"Found {len(files)} coarse files")

# Group files by observation
groups = {}

for f in files:
    base = f.name.rsplit("_coarse_", 1)[0]

    # Keep cadence directories separate
    key = (f.parent.name, base)

    groups.setdefault(key, []).append(f)

print(f"Found {len(groups)} observations")

# Merge each observation
for (cadence, base), coarse_files in sorted(groups.items()):

    outdir = OUTPUT_DIR / cadence
    outdir.mkdir(exist_ok=True)

    outfile = outdir / f"{base}_combined.dat"

    print("\n==============================")
    print(base)
    print(f"{len(coarse_files)} coarse files")

    first_file = True
    total_hits = 0

    with open(outfile, "w") as fout:

        for f in sorted(coarse_files):

            # Skip empty BLISS outputs
            if f.stat().st_size == 0:
                continue

            with open(f, "r") as fin:

                for line in fin:

                    # Keep the full header from the first file only
                    if first_file:
                        fout.write(line)

                    else:
                        # Skip comment/header lines in later files
                        if line.startswith("#"):
                            continue
                        fout.write(line)

                    # Count data rows
                    if not line.startswith("#") and line.strip():
                        total_hits += 1

            first_file = False

    print(f"Saved: {outfile}")
    print(f"Hits: {total_hits}")

print("\nDone!")