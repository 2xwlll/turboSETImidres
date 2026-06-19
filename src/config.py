from pathlib import Path


class Config:
    def __init__(self):
        self.base_dir = Path("/datax/scratch/wlll2x")

        self.data_dir = self.base_dir / "raw"
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"

        self.default_fil = self.data_dir / "voyager_f1032192_t300_v2.fil"

        self.plot_waterfall = True

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)