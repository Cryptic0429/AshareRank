# AshareRank Research Plan and Status

## Working question

For A-share cross-sectional ranking, does a sequence/cross-sectional model such as StockMixer improve five-day excess-return ranking over simpler baselines (at minimum a linear baseline and LightGBM), and does any improvement persist in a cost-adjusted portfolio evaluation?

This is a working scope, not a completed or preregistered scientific claim. Freeze the universe, period, labels, and primary metric before a confirmatory run.

## Hypotheses

- **H1 — Predictive ranking:** the research model improves mean out-of-sample RankIC over the strongest baseline on the same observations.
- **H2 — Stability:** the improvement is not isolated to one period or random seed.
- **H3 — Economic relevance:** the improvement remains directionally positive after turnover and transaction costs are included.

These hypotheses must be evaluated separately. Better RankIC does not by itself imply a profitable strategy.

## Proposed protocol

1. Define the point-in-time A-share universe and a fixed observation period. Prefer historical index constituents; document delisted names and missing-history handling.
2. Build features from information available by the prediction date. Define the five-trading-day forward excess-return label before splitting.
3. Use chronological walk-forward validation. Purge at least five trading dates at each boundary where label windows could overlap; document all fold dates and exclusions.
4. Fit preprocessing only on each training fold. Use validation data for hyperparameter selection and early stopping.
5. Compare the same baseline and research models on the identical test dates and common `(date, ticker)` coverage.
6. Report RankIC as the proposed primary predictive metric; report IC, ICIR, Precision@K, and uncertainty as supporting results.
7. Report portfolio return, Sharpe, maximum drawdown, turnover, and net results under predeclared cost scenarios. Keep strategy assumptions separate from predictive metrics.
8. Run robustness checks across time windows, seeds, universe definitions, label horizons, and transaction costs before making broad claims.

## Data provenance

The existing provider layer supports Tushare and AkShare, with local caching and synthetic fallback. Each empirical run must record provider, retrieval time, covered date range, missingness, adjustment convention, universe constituents, and whether any rows came from fallback or synthetic data. The current built-in synthetic return generator is now seeded with a stable digest, but synthetic data remains a software-development aid rather than empirical evidence.

## Run record

Each run should be identifiable by a unique run ID and preserve:

- exact CLI command and resolved config;
- Git revision and dirty/clean state;
- Python and dependency versions;
- data manifest and checksums;
- random seeds and model checkpoints;
- split boundaries, purge dates, and sample counts;
- per-date predictions, metrics, uncertainty estimates, and report files.

Generated data and runs are ignored by Git. Commit only code, configs, protocol changes, and curated reports with traceable run IDs.

## Current state

- **Repository direction:** research workflow; no active game, web API, or presentation UI.
- **Core retained:** data providers, panel construction, features and labels, training adapters, baseline and deep models, alpha evaluation, and portfolio risk calculations.
- **Current primary benchmark:** `scripts/research/run_benchmark.py` compares StockMixer, LightGBM, XGBoost, and LSTM using a shared evaluation layer.
- **Known work before confirmatory claims:** distinguish historical constituents from current-universe backfills; complete provider-level provenance for every model branch; define and freeze one primary metric and cost assumptions.
- **Refactor status:** the shared chronological split, primary four-model benchmark, and secondary rolling LightGBM/XGBoost/LSTM workflows purge five trading dates at the train/validation boundaries. Rolling reports now retain fold boundaries and excluded dates. Complete provider-level provenance remains open because existing local OHLCV caches do not consistently identify their original source.
- **Experiment status:** no new training run has been launched as part of this refactor.
- **Setup status:** the README documents the Windows PowerShell setup and optional `data`, `tree`, `deep`, and `dev` dependency groups; installation has not been exercised in this refactor.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-09-23 | Make the repository a research codebase and remove the game, Streamlit UI, and FastAPI shell from the active project. | Focus on reproducibility and evidence rather than product presentation. |
| 2026-09-23 | Use five-day A-share excess-return ranking as the working research question. | It connects the existing feature, model-comparison, and evaluation code into one testable line of inquiry. |
