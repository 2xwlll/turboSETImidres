from pathlib import Path
import pandas as pd

csv_file = (
    "/home/wlll2x/midres_narrow/turboSETI/data/carmen/"
    "carmenes_1688mhz_subset.csv"
)

output_dir = Path(
    "/datax/scratch/wlll2x/carmen_midres"
)

output_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(csv_file)

# column containing fine resolution files
paths = df[".h5 path"]

# convert to midres paths
midres_paths = []

for p in paths:

    p = Path(p)

    if "gpuspec.0000.h5" in p.name:
        new = p.with_name(
            p.name.replace(
                "gpuspec.0000.h5",
                "gpuspec.0002.h5"
            )
        )

    elif "rawspec.0000.h5" in p.name:
        new = p.with_name(
            p.name.replace(
                "rawspec.0000.h5",
                "rawspec.0002.h5"
            )
        )

    else:
        raise ValueError(p)

    midres_paths.append(str(new))


# Group every six observations
for i in range(0, len(midres_paths), 6):

    group_num = i // 6

    group_dir = output_dir / f"cadence_{group_num:03d}"
    group_dir.mkdir(exist_ok=True)

    group = df.iloc[i:i+6].copy()

    files = midres_paths[i:i+6]


    with open(group_dir/"all.lst","w") as f:
        for x in files:
            f.write(x+"\n")


    # assuming:
    # even indices = ON
    # odd indices = OFF

    with open(group_dir/"on.lst","w") as f:
        for x in files[0::2]:
            f.write(x+"\n")


    with open(group_dir/"off.lst","w") as f:
        for x in files[1::2]:
            f.write(x+"\n")


print("Done!")