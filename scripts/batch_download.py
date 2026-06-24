from pathlib import Path
import requests

outdir = Path("/datax/scratch/wlll2x/raw/voyager")
outdir.mkdir(parents=True, exist_ok=True)

with open("voyager_urls.txt") as f:
    urls = [line.strip() for line in f if line.strip()]

for url in urls:
    fname = outdir / url.split("/")[-1]

    if fname.exists():
        print("Skipping", fname.name)
        continue

    print("Downloading", fname.name)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f_out:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f_out.write(chunk)