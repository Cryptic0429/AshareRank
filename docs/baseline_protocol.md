# Frozen simple-model experiment, 2026-09-28

This is an exploratory empirical comparison requested by the user. StockMixer and LSTM are deferred. Configuration is frozen in `configs/public_baselines.json` before viewing any test results.

## Data

Use BaoStock historical CSI300 membership requested as of 2018-01-02. Select 60 members by ascending SHA256 of the fixed seed plus stock code. Do not replace failed or delisted constituents based on subsequent performance. This is a fixed historical cohort, not a dynamic index. Download 2017–2025 adjusted daily bars (`adjustflag=1`, backward adjustment / 后复权) with historical trading and ST flags. Archive the complete membership response and trading calendar. Each ticker has retrieval time, requested and covered dates, null counts, endpoint and SHA256. Reuse a file only if its checksum agrees with its sidecar. Abort incomplete cohort downloads; never synthesize or silently switch providers.

## Signals, labels and splits

The information cutoff is signal-date close. Signal-time eligibility requires 60 observed closes, non-ST trading status, and at least CNY 10 million traded that day. Sixteen price/volume features use only current/past observations, followed by same-date cross-sectional ranks over the eligible cohort. Imputation and scaling are fitted on training rows only.

The label return is `open[t+6] / open[t+1] - 1`: five sessions held, entering at the next open. Subtract the same-date equally weighted observed eligible-cohort return for the excess-return metric. Models fit the cross-sectional rank of the label, not calibrated expected returns. Preprocessing must not use future outcome availability to decide which stocks enter the signal universe.

Test years are 2023, 2024, 2025. For each test year, the previous calendar year is validation and all earlier dates from 2018-01-02 are training. Construct labels first and retain training/validation rows only if their label-end date is strictly earlier than the next split. Record actual excluded dates. Do not tune on test results. All models share the same observations, features and target; common observed labels form the predictive evaluation intersection. Keep full signal-time predictions, including missing future outcomes, for portfolio selection.

## Models and uncertainty

Ridge, Elastic Net, Random Forest, LightGBM and XGBoost each have two predeclared candidate configurations. Choose the largest mean validation RankIC at seed 17; fixed iteration limits avoid test-set early stopping. Average seeds 17, 29 and 43 for tree-model headline predictions, and retain individual seed results. Linear models are deterministic. Twenty-day positive momentum is a fixed, untuned reference.

Report daily IC, RankIC, unannualized RankICIR, Precision@10 relative to positive cohort-excess returns, coverage and base rate. For mean RankIC and paired differences vs Ridge, use a 20-date circular moving-block bootstrap with 1,000 resamples and a fixed seed. These are exploratory, unadjusted intervals, not multiplicity-controlled confirmation.

## Portfolio proxy

Top 10 equal weights; next-open execution every five exchange-calendar sessions. EqualWeight holds all signal-eligible stocks with the same execution and costs. Buy plus sell notional divided by pre-trade equity is two-way turnover. Test 0, 10, 25 and 50 bps per traded side and all five rebalance-start offsets at 10 bps. Model cash and fees explicitly; do not borrow to pay fees. Include initial entry and attempted final liquidation. Value holdings daily at adjusted opens. Report daily-return Sharpe (zero risk-free rate, 252 sessions), CAGR, maximum drawdown including initial capital, blocked orders and stale-price holding days.

Suspended/missing-price orders cannot execute. A conservative +/-9.5% next-open gap screen blocks buys/sells, respectively. This is an approximation, not the exact historical exchange limit regime. Blocked sales remain invested; blocked buys stay in cash. Missing held prices use last available marks solely for accounting and never for execution. Unknown delisting/merger proceeds and remaining terminal holdings are material limitations, not zero-risk cash. There is no bid/ask, market impact, participation cap or round-lot model. Net results are simulation proxies, not tradable performance claims.

## Reproduction

Install `python -m pip install -e ".[research]"`, then run `python scripts/research/run_public_baselines.py --config configs/public_baselines.json`. `--download-only` acquires/resumes verified snapshots. Raw snapshots, models, predictions and manifests remain under ignored `data/` and `runs/`. Publish only curated bilingual reports, aggregate summaries and figures under `reports/baselines/`. Every run records the exact command, config, seeds, environment, revision, dirty status, source hashes, input checksums and split exclusions.

New deterministic checks are supplied in `tests/research/test_public_baselines.py`; project policy requires an explicit request before executing tests. Downloading, training and evaluation are authorized by the current user request.
