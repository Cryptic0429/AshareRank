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
