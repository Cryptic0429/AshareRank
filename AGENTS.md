# AshareRank Research — Agent Instructions

This file governs work in this repository. Read it together with `RESEARCH_PLAN.md` before changing research code.

## Project objective

Maintain a reproducible research codebase for A-share cross-sectional stock ranking. The working research question is whether a more complex model improves out-of-sample ranking of five-day excess returns over credible baselines, and whether the improvement remains after costs.

This is research software, not a product. Do not add a game, dashboard, API service, or product-oriented UI unless the user explicitly changes this objective.

## Architecture boundaries

- `src/asharerank/data_service.py` and `src/asharerank/settings.py`: provider access, local data paths, and runtime settings.
- `src/asharerank/data/`, `forecasting/`, `models/`, `adapters/`, and `training/`: dataset construction, forecast experiments, and model fitting.
- `src/asharerank/prediction/`: prediction contracts, ranking, selection, and inference engines.
- `src/asharerank/evaluation/` and `src/asharerank/risk/`: predictive metrics, portfolio evaluation, and risk analysis.
- `scripts/research/`: reproducible command-line entry points. Keep orchestration thin; move reusable logic into `src/asharerank/`.
- `configs/` (to be introduced when the CLI consumes file-based configs): committed experiment definitions. Never embed private credentials in configs.
- `data/` and `runs/`: generated local artifacts. Do not commit raw data, caches, model weights, predictions, logs, or machine-specific outputs.
- `reports/`: curated findings only. Every result must point to its run identifier and assumptions.
- `tests/research/`: deterministic checks for data transformations, labels, split boundaries, metrics, and serialization.

## Research integrity requirements

1. Preserve point-in-time semantics. A feature must only use data that would have been available at its prediction date.
2. Build labels before splitting, then purge dates whose forward-label windows overlap a later split. Keep an explicit record of excluded dates and horizon.
3. Fit imputers, scalers, feature selection, and other learned transforms on training data only. Apply the fitted transforms to validation and test data.
4. Keep model selection and early stopping within training/validation data. Do not tune against the final test set.
5. Use identical universe, dates, labels, and sample intersection when comparing models. Report both predictive metrics and cost-adjusted portfolio metrics.
6. Record the data manifest, configuration, random seeds, dependency versions, Git revision, and command for each experiment.
7. Label real, cached, fallback, and synthetic data distinctly. Synthetic data is useful for software checks but is not empirical evidence.
8. Never describe percentile-mapped scores as predicted returns or calibrated confidence.
9. Report sample counts, missing coverage, transaction costs, turnover, uncertainty, and limitations with headline metrics.

## Change workflow

- Keep changes focused on the research workflow and preserve existing data/model logic unless the user requests scientific changes.
- Update `RESEARCH_PLAN.md` when the research question, protocol, risks, or project status changes.
- Update `docs/research_protocol.md` when experiment standards change.
- Add or update tests when changing scientific transformations or metric definitions. Do not weaken a test merely to make an implementation pass.
- Do not run tests unless the user asks to test or verify the implementation. If tests are run at the user's request, report the exact command and outcome.
- Do not run long data downloads or training jobs without an explicit user request; they may incur provider limits and consume substantial compute.
- Before concluding work, inspect the Git diff and identify unverified assumptions.

## Current migration status

- [x] Research-first repository layout and package namespace created.
- [x] Game, Streamlit UI, and FastAPI shell removed from the active tree; local originals retained under ignored `.legacy-archive/`.
- [x] Main research commands moved under `scripts/research/`.
- [x] Add horizon-aware purging to the shared chronological split helper and the four-model primary benchmark paths.
- [x] Audit all training and benchmark split implementations for horizon-aware purging; the shared rolling-window workflows now purge and record boundary dates.
- [x] Separate forecast research workflows from prediction contracts and model-inference runtime namespaces.
- [x] Add versioned run reports, environment manifests, and full CLI argument snapshots to the primary multi-model benchmark.
- [x] Extend consistent lifecycle manifests to training/benchmark entry points, and input provenance to shared loaders and report writers. Unknown legacy provenance remains explicitly unknown; the new historical-cohort baseline workflow has complete per-ticker snapshot records.
- [x] Document the Windows setup and optional dependency installation paths.
- [x] Reconcile the active project description and migration tracker with the research-only repository.

Data lineage remains an open item: existing local OHLCV Parquet files do not consistently retain which provider supplied each ticker or whether a cached file was previously refreshed from a fallback source. Runs using these files must identify that limitation instead of implying complete source provenance.

The first empirical simple-model run is `20260928T065633Z_simple_baselines`; see its curated report and `docs/baseline_protocol.md`. It uses real BaoStock historical-cohort data and no synthetic fallback. StockMixer and LSTM are deferred while simple models are prioritized. Preserve the distinction between the completed empirical workflow and unresolved data-quality limitations (historical revisions, delisting/merger settlement, and assumed fundamental publication dates).
