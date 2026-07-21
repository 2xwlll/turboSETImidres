import h5py
import os
import glob

indir = "/datax/scratch/wlll2x/mars_midres"
outdir = "/datax/scratch/wlll2x/mars_midres_coarse968"

os.makedirs(outdir, exist_ok=True)

coarse = 968

start = coarse * 1024
stop = (coarse + 1) * 1024


for filename in glob.glob(indir+"/*.h5"):

    print("Processing:", os.path.basename(filename))

    outfile = os.path.join(
        outdir,
        os.path.basename(filename).replace(
            ".h5",
            "_coarse968.h5"
        )
    )

    with h5py.File(filename,"r") as fin:
        with h5py.File(outfile,"w") as fout:

            # copy root attributes
            for key,val in fin.attrs.items():
                fout.attrs[key]=val

            # crop data
            d = fout.create_dataset(
                "data",
                data=fin["data"][:,:,start:stop],
                dtype="float32"
            )

            # copy data attributes
            for key,val in fin["data"].attrs.items():
                d.attrs[key]=val

            # update metadata
            d.attrs["nchans"] = 1024
            d.attrs["fch1"] = (
                fin["data"].attrs["fch1"]
                +
                start*fin["data"].attrs["foff"]
            )

            # crop mask
            m = fout.create_dataset(
                "mask",
                data=fin["mask"][:,:,start:stop]
            )

            for key,val in fin["mask"].attrs.items():
                m.attrs[key]=val


    print("Saved:", outfile)