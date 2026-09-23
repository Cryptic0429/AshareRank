# Research architecture

AshareRank is a Python research package with command-line experiment orchestration. There is no active web server or product UI.

```text
scripts/research/ or `asharerank` CLI
            │
            ▼
       src/asharerank
  ┌─────────┼──────────────┐
  data   features/labels  models/training
  │                         │
  └─────── evaluation ◄─────┘
                  │
                  ▼
      runs/<experiment-id>/
      config + manifest + predictions + metrics + report
```

## Package responsibilities

- `settings.py`: project-relative paths, provider credentials, and shared research defaults.
- `data_service.py`, `predict/universe.py`, `predict/fundamentals.py`: provider access, local cache, stock-universe data, and optional fundamentals.
- `data/`, `features/`, `predict/dataset.py`: input validation, feature construction, forward labels, and prepared panels.
- `data/splits.py` and `predict/dataset.py`: chronological and rolling train/validation/test assignment. The shared chronological path, primary benchmark paths, StockMixer/LSTM paths, and secondary rolling LightGBM/XGBoost/LSTM workflows purge five trading dates at each applicable label boundary and record the excluded dates.
- `models/`, `adapters/`, `training/`, and `predict/experimental/`: baseline and research models, training, inference, and benchmark orchestration.
- `evaluation/`, `predict/evaluate.py`, `risk/`: cross-sectional ranking metrics, cost-aware portfolio analysis, and risk decomposition.
- `db.py`: small local SQLite store for risk metric history; experiment artifacts belong in versioned run directories.

## Data and artifact flow

Provider downloads and derived panels are kept under ignored `data/`. Model weights, temporary benchmark frames, run reports, and logs are stored under ignored `runs/`. Commit only source, configs, protocol, tests, and curated reports. A report intended to support a claim must reference its run ID and data manifest.

Live providers are Tushare and AkShare, with local cache and synthetic fallback. Synthetic data must not be presented as empirical evidence. The data layer's current process-stable synthetic seed is for reproducible development behavior only.

The primary four-model benchmark saves a versioned report, full argument snapshot, environment manifest, model predictions, common-universe predictions, and an append-only run index under `runs/experiments/`. Secondary rolling training reports include their fold boundaries and purged dates. Provider-level provenance remains incomplete for existing local OHLCV caches; see `RESEARCH_PLAN.md` before treating a run as confirmatory.

## Interfaces

Use the installed `asharerank` CLI for the supported workflows. The command delegates to focused scripts in `scripts/research/`; reusable research code stays in `src/asharerank/`. Notebooks may be used for exploration but should not be the only place where a final result can be reproduced.
