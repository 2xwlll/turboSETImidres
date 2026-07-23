from pathlib import Path
import pandas as pd

# ==========================================================
# Input / Output
# ==========================================================

csv_file = (
    "/home/wlll2x/midres_narrow/turboSETI/data/carmen/"
    "carmenes_1688mhz_subset.csv"
)

output_lst = (
    "/home/wlll2x/midres_narrow/carmen_midres/"
    "data/carmenes_midres.lst"
)

# ==========================================================
# Read CSV
# ==========================================================

df = pd.read_csv(csv_file)

# Column containing fine-resolution path
fine_paths = df.iloc[:,5]

midres_files = []
missing = []

print("="*70)
print("Searching for corresponding rawspec.0002.h5 files")
print("="*70)


# ==========================================================
# Search
# ==========================================================

for i, fine_file in enumerate(fine_paths, start=1):

    fine_file = Path(fine_file)
    directory = fine_file.parent

    if "gpuspec.0000.h5" in fine_file.name:

        expected = fine_file.name.replace(
            "gpuspec.0000.h5",
            "gpuspec.0002.h5"
        )

    elif "rawspec.0000.h5" in fine_file.name:

        expected = fine_file.name.replace(
            "rawspec.0000.h5",
            "rawspec.0002.h5"
        )

    else:

        print("\nWARNING: Unknown filename type:")
        print(fine_file.name)
        missing.append(str(fine_file))
        continue

    candidate = directory / expected

    print(f"\n[{i}/{len(fine_paths)}]")
    print(f"Fine : {fine_file.name}")
    print(f"Look : {candidate.name}")

    if candidate.exists():

        print("✓ Found")
        midres_files.append(str(candidate))

    else:

        print("✗ Couldn't find a match")
        missing.append(str(candidate))


# ==========================================================
# Remove duplicates
# ==========================================================

before = len(midres_files)

midres_files = sorted(set(midres_files))

duplicates_removed = before - len(midres_files)


# ==========================================================
# Save .lst
# ==========================================================

with open(output_lst,"w") as f:

    for filename in midres_files:
        f.write(filename+"\n")


# ==========================================================
# Summary
# ==========================================================

print("\n")
print("="*70)
print("Summary")
print("="*70)

print(f"Rows in CSV          : {len(fine_paths)}")
print(f"Matches found        : {len(midres_files)}")
print(f"Missing              : {len(missing)}")
print(f"Duplicates removed   : {duplicates_removed}")

print("\nSaved list:")
print(output_lst)


if missing:

    print("\nExample missing files:")
    
    for filename in missing[:20]:
        print(filename)

    if len(missing) > 20:
        print(
            f"... plus {len(missing)-20} more"
        )