"""Reproducible historical-cohort baseline comparison, independent of StockMixer."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/public_baselines.json')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    from asharerank.data.public_history import acquire
    manifest = acquire(config, ROOT)
    print(f'Data manifest: {manifest}', flush=True)
    if not args.download_only:
        from asharerank.forecasting.simple_benchmark import run
        print(run(config, manifest, ROOT), flush=True)


if __name__ == '__main__':
    main()
