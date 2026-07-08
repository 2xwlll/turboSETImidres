from pathlib import Path
from blimpy import Waterfall

directory = Path("/datag/public/A_Sign_in_Space_raw_h5")

patterns = [
    "*.rawspec.0000.h5",
    "*.rawspec.0001.h5",
    "*.rawspec.0002.h5",
]

for pattern in patterns:
    matches = sorted(directory.glob(pattern))
    if not matches:
        continue

    f = matches[0]

    print("=" * 80)
    print(f"Inspecting: {f.name}")

    # Header only
    wf = Waterfall(str(f), load_data=False)

    print(f"foff   : {wf.header['foff']}")
    print(f"tsamp  : {wf.header['tsamp']}")
    print(f"nchans : {wf.header['nchans']}")
    print(f"shape  : {wf.container.selection_shape}")

    # Read only the first 10 frequency channels
    fch1 = wf.header["fch1"]
    foff = wf.header["foff"]

    wf_small = Waterfall(
        str(f),
        f_start=fch1 + 10 * foff,
        f_stop=fch1,
    )

    print("\nTiny data sample:")
    print(wf_small.data)
    print()

from pathlib import Path

directory = Path("/datag/public/A_Sign_in_Space_raw_h5")

midres_files = sorted(directory.glob("*.rawspec.0002.h5"))

print(f"Found {len(midres_files)} mid-resolution files:\n")

for f in midres_files:
    print(f.name)

from blimpy import Waterfall

wf = Waterfall(
    "/datag/public/A_Sign_in_Space_raw_h5/blc33_guppi_60067_75323_PSR_B0355+54_0012.rawspec.0002.h5",
    load_data=False,
)

print(wf.header["fch1"])
print(wf.header["foff"])
print(wf.header["nchans"])