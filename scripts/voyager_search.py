# scripts/run_voyager.py

from src.run_manager import RunManager


FILE = "/datax/scratch/wlll2x/voyager_f1032192_t300_v2.fil"


def main():
    runner = RunManager()
    wf = runner.process_file(FILE)

    print("\nPipeline complete")


if __name__ == "__main__":
    main()