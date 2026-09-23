# AshareRank Project Tracker

## Project name

**AshareRank** — a concise name for A-share cross-sectional stock-ranking research. The Python package and command-line tool use the lowercase name `asharerank`.

## Objective

Maintain a reproducible research project studying whether sequence and cross-sectional models improve out-of-sample ranking of five-day excess returns, and whether any predictive improvement remains useful after portfolio costs.

## Current pipeline

```text
Tushare / AkShare / local cache
    → validated A-share panel and universe
    → point-in-time features and forward labels
    → chronological splits with label-window purge
    → StockMixer / LightGBM / XGBoost / LSTM
    → shared-sample ranking and portfolio evaluation
    → versioned experiment archive under runs/experiments/
```

Main entry points: `asharerank benchmark`, `asharerank build-dataset`, and `asharerank list-experiments`.

## Milestones

- [x] Move research code into `src/asharerank/` and command-line workflows into `scripts/research/`.
- [x] Keep game, web UI, and API shell outside the active project tree.
- [x] Add horizon-aware purging and per-model split metadata to the primary four-model benchmark.
- [x] Save versioned benchmark reports, run arguments, environment details, and predictions.
- [x] Rename package, CLI, project metadata, and documentation to AshareRank.
- [x] Audit secondary chronological training workflows for five-day label-window purging and record the excluded dates.
- [ ] Complete data-source provenance and point-in-time universe documentation; older local Parquet caches do not identify their original provider consistently.
- [ ] Freeze universe, time period, primary metric, and cost assumptions before confirmatory runs.
- [x] Document the Windows environment setup and optional provider/model dependency groups.

## Research safeguards

Treat benchmark results as exploratory until the universe history, feature availability, data provenance, and robustness checks are documented. Compare models on the same date/ticker intersection; use validation for model selection; clearly identify cached, fallback, and synthetic observations. Synthetic data is not empirical evidence.

## Project records

- Research question, protocol, decisions, and status: `RESEARCH_PLAN.md`
- Coding and research integrity rules: `AGENTS.md`
- Architecture: `docs/architecture.md`
- Evaluation protocol: `docs/research_protocol.md`
