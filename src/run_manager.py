# src/run_manager.py

from src.config import Config
from src.data_io import load_data
from src.search import run_search
from src.candidates import postprocess_candidates
from src.utils import setup_logger


class RunManager:
    def __init__(self, config):
        self.config = config
        self.config.ensure_dirs()

        self.logger = setup_logger(
            "run_manager",
            self.config.logs_dir / "run.log"
        )

    def run(self, input_path):
        self.logger.info("Starting pipeline run")

        wf = load_data(input_path, self.config)

        self.logger.info(f"Data shape: {wf.data.shape}")

        raw_candidates = run_search(wf, self.config)

        self.logger.info(f"Raw candidates: {len(raw_candidates)}")

        final_candidates = postprocess_candidates(raw_candidates, self.config)

        self.logger.info(f"Final candidates: {len(final_candidates)}")

        return final_candidates