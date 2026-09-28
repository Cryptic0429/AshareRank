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

- **Repository direction:** research workflow, with a separate bilingual public project presentation in `website/`; no active game or web API.
- **Core retained:** data providers, panel construction, features and labels, training adapters, baseline and deep models, alpha evaluation, and portfolio risk calculations.
- **Current empirical priority:** `scripts/research/run_public_baselines.py` compares Ridge, Elastic Net, Random Forest, LightGBM and XGBoost against momentum on a frozen historical cohort. StockMixer and LSTM are deferred. The older four-model benchmark remains available.
- **Known work before confirmatory claims:** distinguish historical constituents from current-universe backfills; complete provider-level provenance for every model branch; define and freeze one primary metric and cost assumptions.
- **Refactor status:** the shared chronological split, primary four-model benchmark, and secondary rolling LightGBM/XGBoost/LSTM workflows purge five trading dates at the train/validation boundaries. Rolling reports now retain fold boundaries and excluded dates. Complete provider-level provenance remains open because existing local OHLCV caches do not consistently identify their original source.
- **Package layout:** forecast dataset/model/benchmark workflows now live under `asharerank.forecasting`; prediction contracts and inference engines share `asharerank.prediction`, with model runtime code under `asharerank.prediction.inference`.
- **Experiment status:** completed run `20260928T065633Z_simple_baselines`, source revision `4a501c1da1f5ea38769fcb3ae20260cb695bc271` (clean). BaoStock supplied 130,505 real daily rows for 60 deterministically selected 2018-01-02 CSI300 constituents. Test years 2023–2025 yield 41,456 common observations over 709 dates. Curated bilingual results and four figures are in `reports/baselines/20260928T065633Z_simple_baselines/`.
- **Observed result:** Elastic Net has the largest mean RankIC (0.0571; 95% moving-block interval 0.0186–0.0925), versus Ridge 0.0203, LightGBM 0.0274 and XGBoost 0.0271. At 10 bps per traded side, its portfolio proxy CAGR is 13.7%, Sharpe 0.82 and maximum drawdown -21.9%. These are exploratory fixed-cohort findings; cost and start-offset sensitivity, year/seed variation, and incomplete delisting settlement prohibit broad profitability claims.
- **Provenance status:** training/benchmark command entry points now have common lifecycle records; model reports and dataset exports record registered input hashes and explicit unknown provenance. Fresh legacy provider downloads retain row-level provider/time/adjustment fields. The new empirical workflow has immutable per-ticker lineage and rejects incomplete downloads. Historical cache lineage cannot be retroactively recovered.
- **Validation status:** the authorized data acquisition, training and report-generation command completed successfully; deterministic tests were added but not executed under the repository's explicit-testing-request rule. Legacy pathways with assumed fundamental announcement dates remain unsuitable for point-in-time claims.
- **Setup status:** the README documents the Windows PowerShell setup and optional `data`, `tree`, `deep`, and `dev` dependency groups; installation has not been exercised in this refactor.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-09-23 | Make the repository a research codebase and remove the game, Streamlit UI, and FastAPI shell from the active project. | Focus on reproducibility and evidence rather than product presentation. |
| 2026-09-23 | Use five-day A-share excess-return ranking as the working research question. | It connects the existing feature, model-comparison, and evaluation code into one testable line of inquiry. |
| 2026-09-28 | Add a Chinese/English public project website and publish the repository on GitHub at the user's request. | Explain the research question, methodology and open limitations without presenting unconfirmed performance claims or changing the scientific workflow. |
| 2026-09-28 | Prioritize simple baselines and use a fixed historical BaoStock cohort with next-open five-session labels. | Fulfill the requested empirical comparison without requiring StockMixer or backfilling current constituents. Freeze hyperparameters, costs, seeds and dates before inspecting test outcomes. |
