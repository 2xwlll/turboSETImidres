#!/usr/bin/env python3

import sys
import h5py

fname = sys.argv[1]

with h5py.File(fname, "r") as f:

    data = f["data"]

    print("\nDataset")
    print("-------")
    print("shape :", data.shape)
    print("dtype :", data.dtype)

    print("\nAttributes")
    print("----------")

    for k, v in sorted(data.attrs.items()):
        print(f"{k:20s} : {v}")