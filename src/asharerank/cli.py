"""Small command-line entry point for reproducible research workflows."""
from __future__ import annotations

import argparse
import runpy
import sys

from asharerank.settings import PROJECT_ROOT

_COMMANDS = {
    "fetch-universe": "fetch_universe.py",
    "fetch-index": "fetch_index.py",
    "fetch-fundamentals": "fetch_fundamentals.py",
    "build-dataset": "build_dataset.py",
    "benchmark": "run_benchmark.py",
    "public-baselines": "run_public_baselines.py",
    "train-models": "train_models.py",
    "train-stockmixer": "train_stockmixer.py",
    "train-lstm": "train_lstm.py",
    "train-xgboost": "train_xgboost.py",
    "list-experiments": "list_experiments.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="asharerank",
        description="Run data and model research workflows. Use '<command> --help' for options.",
    )
    parser.add_argument("command", choices=sorted(_COMMANDS))
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args([argv[0]])
    script = PROJECT_ROOT / "scripts" / "research" / _COMMANDS[args.command]
    if not script.is_file():
        parser.error(f"workflow script not found: {script}")
    # Preserve each workflow script's existing argparse interface.
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
