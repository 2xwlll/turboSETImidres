from turbo_seti.find_doppler.find_doppler import FindDoppler

DATAFILE = "/datax/scratch/wlll2x/raw/voyager_f1032192_t300_v2.fil"
OUTPUT_DIR = "/datax/scratch/wlll2x/results"

fd = FindDoppler(
    datafile=DATAFILE,
    out_dir=OUTPUT_DIR,
    min_drift=-5,
    max_drift=5,
    snr=10
)

fd.search()