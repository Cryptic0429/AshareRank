# Research protocol

## Dataset definition

Record the market, date range, universe membership rule, corporate-action adjustment, delisting and suspension handling, feature availability time, forward-return horizon, and missing-data treatment. Prefer historical point-in-time universe membership. Keep raw source snapshots immutable and write normalized data to a separate processed location.

## Splits and leakage control

Split by date, never by individual ticker rows. For a forward label with horizon `h`, remove at least the final `h` eligible observation dates before each later split boundary from the earlier partition. Report the boundary dates and purged dates. For overlapping labels or longer feature look-ahead, use an embargo appropriate to the full information window.

Fit all learned preprocessing (imputation, scaling, feature selection, calibration) on training data only. Select models and stop training with validation data only. Do not use the final test set to choose features, models, or thresholds.

## Fair model comparisons

Use the same universe, prediction dates, target, information set, and trading assumptions. Report the common `(date, ticker)` intersection as the primary comparison sample; report individual coverage separately. Preserve predictions at the sample level so metrics can be independently recomputed.

## Metrics

### Predictive

- IC and RankIC by date, with mean, dispersion, and sample count.
- ICIR and RankICIR with their exact aggregation definition stated.
- Precision@K and the positive-return base rate for context.
- Confidence intervals or resampling-based uncertainty for headline comparisons.

### Portfolio simulation

- Return, Sharpe, maximum drawdown, and turnover.
- Gross and net metrics under explicit transaction-cost scenarios.
- Rebalancing cadence, holding period, liquidity filters, and treatment of suspensions/limits.

Predictive skill and portfolio profitability are distinct claims and must be reported separately.

## Robustness and reporting

Check alternative time windows, seeds, universe definitions, cost levels, and reasonable label horizons. Keep exploratory and confirmatory runs distinguishable. For each reported result include run ID, source and data version, code revision, environment, config, split/purge dates, metrics, sample coverage, limitations, and exact reproduction command.

Never describe a rank-percentile transform as a predicted return or calibrated confidence. Mark cached, fallback, and synthetic data clearly.

## Recorded command lifecycle

Training, benchmark, dataset-building and universe/fundamental-download entry points now use a shared command recorder. Each invocation writes an initial and final manifest under `runs/commands/<run_id>/`, including failures, full command, resolved CLI arguments, package versions, Git revision and source-file hashes. Model reports and panel artifacts also carry environment and registered input manifests. Instrumented market and fundamental loaders register file hashes, coverage, missingness and known row-level provenance. The input list describes files read by the process, potentially a superset for one model; it does not claim to discover uninstrumented external data.

Fresh legacy OHLCV downloads retain exact provider endpoint, retrieval time, adjustment and data kind on every row. Old rows without this information stay unknown after merging. A legacy price cache without a `.src` file is unknown, never presumed to be AkShare. Retrospective provenance cannot be reconstructed from file names. Legacy fundamental availability estimated as report-date plus 45 days is not a verified publication timestamp and cannot support point-in-time claims.

The completed simple-model study uses a separate immutable BaoStock snapshot with a historical fixed cohort and no synthetic/fallback rows. Its frozen protocol, next-open label, six-session boundary requirement, cost scenarios, uncertainty method and execution limitations are specified in [`baseline_protocol.md`](baseline_protocol.md). This study is exploratory; StockMixer is deferred.
