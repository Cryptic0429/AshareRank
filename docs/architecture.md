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
- `data_service.py`, `forecasting/universe.py`, `forecasting/fundamentals.py`: provider access, local cache, stock-universe data, and optional fundamentals.
- `data/`, `forecasting/dataset.py`, `forecasting/features.py`: input validation, feature construction, forward labels, and prepared panels.
- `data/splits.py` and `forecasting/dataset.py`: chronological and rolling train/validation/test assignment. The shared chronological path, primary benchmark paths, StockMixer/LSTM paths, and secondary rolling LightGBM/XGBoost/LSTM workflows purge five trading dates at each applicable label boundary and record the excluded dates.
- `forecasting/`: data-backed forecast experiments, model training workflows, benchmark orchestration, and research-only model artifact metadata. Import these modules as `asharerank.forecasting.*`.
- `prediction/`: stable prediction-frame contracts, ranking, and universe/portfolio selection helpers. This package does not own model training.
- `prediction/inference/`: model-loading and scoring engines. `engine.py` scores saved model-comparison artifacts; `service.py` manages that local workflow. `alpha.py` contains the separate older batch scoring and SQLite persistence path.
- `models/`, `adapters/`, `training/`: model definitions, training-data adapters, and the shared model-comparison workflow.
- `evaluation/`, `risk/`: cross-sectional ranking metrics, cost-aware portfolio analysis, and risk decomposition.
- `db.py`: small local SQLite store for risk metric history; experiment artifacts belong in versioned run directories.

The former top-level `predict/` package has been renamed to `forecasting/`. The former top-level `inference/` package now lives under `prediction/inference/`; model-specific runtime helpers and the tree-worker entry point live there as well. This leaves one top-level prediction namespace and makes the distinction between training/benchmark workflows and scoring outputs explicit.

## Data and artifact flow

Provider downloads and derived panels are kept under ignored `data/`. Model weights, temporary benchmark frames, run reports, and logs are stored under ignored `runs/`. Commit only source, configs, protocol, tests, and curated reports. A report intended to support a claim must reference its run ID and data manifest.

Live providers are Tushare and AkShare, with local cache and synthetic fallback. Synthetic data must not be presented as empirical evidence. The data layer's current process-stable synthetic seed is for reproducible development behavior only.

The primary four-model benchmark saves a versioned report, full argument snapshot, environment manifest, model predictions, common-universe predictions, and an append-only run index under `runs/experiments/`. Secondary rolling training reports include their fold boundaries and purged dates. Provider-level provenance remains incomplete for existing local OHLCV caches; see `RESEARCH_PLAN.md` before treating a run as confirmatory.

## Interfaces

Use the installed `asharerank` CLI for the supported workflows. The command delegates to focused scripts in `scripts/research/`; reusable research code stays in `src/asharerank/`. Notebooks may be used for exploration but should not be the only place where a final result can be reproduced.
