"""Historical-cohort, purged walk-forward baselines with an explicit execution proxy."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asharerank.data.public_history import load_snapshot, sha256, write_json


def prepare(panel: pd.DataFrame, dates: pd.DatetimeIndex, config: dict):
    parts, features = [], []
    for code, raw in panel.groupby('code'):
        d = raw.set_index('date').reindex(dates).copy()
        d['date'], d['code'] = dates, code
        for c in ('open', 'close', 'high', 'low'):
            d[c] = d[c].where(d[c] > 0)
        close, volume = d.close, d.volume.where(d.volume > 0)
        daily = close.pct_change(fill_method=None)
        f = pd.DataFrame(index=dates)
        for window in (1, 5, 10, 20, 60):
            f[f'return_{window}'] = close.pct_change(window, fill_method=None)
        for window in (10, 20, 60):
            f[f'volatility_{window}'] = daily.rolling(window, min_periods=window).std()
            f[f'ma_distance_{window}'] = close / close.rolling(window, min_periods=window).mean() - 1
        f['intraday'] = close / d.open - 1
        f['range'] = (d.high - d.low) / close
        f['volume_ratio'] = volume / volume.rolling(20, min_periods=20).mean() - 1
        f['turnover'] = d.turn
        f['turnover_ratio'] = d.turn / d.turn.rolling(20, min_periods=20).mean().replace(0, np.nan) - 1
        features = list(f)
        f['date'], f['code'] = dates, code
        f['eligible'] = (d.tradestatus == 1) & (d.isST == 0) & (d.amount >= config['minimum_amount']) & (close.rolling(60).count() == 60)
        f['raw_return'] = d.open.shift(-(config['horizon'] + 1)) / d.open.shift(-1) - 1
        f['label_end'] = pd.Series(dates, index=dates).shift(-(config['horizon'] + 1))
        parts.append(f.reset_index(drop=True))
    frame = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[frame.eligible].copy()
    # Cross-sectional transforms use the information set at signal time, not future coverage.
    for column in features:
        frame[column] = frame.groupby('date')[column].rank(pct=True) * 2 - 1
    frame['excess_return'] = frame.raw_return - frame.groupby('date').raw_return.transform('mean')
    frame['target'] = frame.groupby('date').raw_return.rank(pct=True) * 2 - 1
    return frame, features


def split_fold(frame: pd.DataFrame, year: int, train_start: str):
    valid_start = pd.Timestamp(year - 1, 1, 1)
    test_start = pd.Timestamp(year, 1, 1)
    test_end = pd.Timestamp(year + 1, 1, 1)
    train_block = frame.loc[(frame.date >= train_start) & (frame.date < valid_start)]
    valid_block = frame.loc[(frame.date >= valid_start) & (frame.date < test_start)]
    test_block = frame.loc[(frame.date >= test_start) & (frame.date < test_end)]
    train = train_block.loc[(train_block.label_end < valid_start) & train_block.target.notna()]
    valid = valid_block.loc[(valid_block.label_end < test_start) & valid_block.target.notna()]
    # Test predictions include missing future outcomes; portfolio selection cannot see labels.
    test = test_block.copy()
    record = {'year': year, 'horizon': 5, 'execution_lag': 1, 'purge_rule': 'label_end strictly before next split boundary', 'train_rows': len(train), 'valid_rows': len(valid), 'test_prediction_rows': len(test)}
    for name, original, kept in [('train', train_block, train), ('valid', valid_block, valid)]:
        record[name + '_dates'] = [str(kept.date.min().date()), str(kept.date.max().date())]
        record[name + '_excluded_dates'] = sorted(set(original.date.dt.strftime('%Y-%m-%d')) - set(kept.date.dt.strftime('%Y-%m-%d')))
    if len(train) == 0 or len(valid) == 0 or len(test) == 0:
        raise ValueError(f'Empty fold {year}')
    return train, valid, test, record


def daily_metrics(frame: pd.DataFrame, column: str, top_k: int, minimum: int):
    records = []
    for date, group in frame.groupby('date', sort=True):
        group = group.dropna(subset=[column, 'excess_return'])
        if len(group) < minimum or group[column].nunique() < 2:
            continue
        rank_ic = group[column].rank().corr(group.excess_return.rank())
        chosen = group.sort_values([column, 'code'], ascending=[False, True]).head(top_k)
        records.append({'date': date, 'rank_ic': rank_ic, 'ic': group[column].corr(group.excess_return), 'precision': (chosen.excess_return > 0).mean(), 'base_rate': (group.excess_return > 0).mean(), 'n': len(group)})
    return pd.DataFrame(records)


def block_interval(values, blocks=20, repetitions=1000, seed=1729):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = (rng.integers(0, len(values), (repetitions, int(np.ceil(len(values) / blocks)), 1)) + np.arange(blocks)) % len(values)
    means = values[indices.reshape(repetitions, -1)[:, :len(values)]].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def estimator(name: str, params: dict, seed: int):
    if name == 'Ridge':
        model = Ridge(**params)
    elif name == 'ElasticNet':
        model = ElasticNet(**params, max_iter=5000, tol=1e-5, random_state=seed)
    elif name == 'RandomForest':
        model = RandomForestRegressor(**params, max_features=0.8, n_jobs=4, random_state=seed)
    elif name == 'LightGBM':
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**params, colsample_bytree=.8, subsample=.8, subsample_freq=1, reg_lambda=1, random_state=seed, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    elif name == 'XGBoost':
        from xgboost import XGBRegressor
        model = XGBRegressor(**params, colsample_bytree=.8, subsample=.8, reg_lambda=1, random_state=seed, n_jobs=4, tree_method='hist', objective='reg:squarederror')
    else:
        raise ValueError(name)
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), model)


def portfolio(predictions: pd.DataFrame, panel: pd.DataFrame, dates: pd.DatetimeIndex,
              column: str, config: dict, cost_bps: float, offset=0):
    """Open-price daily accounting; fixed five-session schedule, no future-label filter.

    Conservative 9.5% open-gap screen for this pre-2018 mainboard/ChiNext cohort.
    Orders blocked by suspension or screen stay in cash / retain existing holdings.
    This is an OHLC execution proxy, not an exchange-accurate fill simulation.
    """
    codes = sorted(panel.code.unique())
    raw = panel.pivot(index='date', columns='code', values='open').reindex(index=dates, columns=codes)
    raw = raw.where(raw > 0)
    marked = raw.ffill()  # valuation only; missing prices can never be used for orders
    close = panel.pivot(index='date', columns='code', values='close').reindex(index=dates, columns=codes)
    status = panel.pivot(index='date', columns='code', values='tradestatus').reindex(index=dates, columns=codes)
    gap = raw / close.shift(1) - 1
    signals = {pd.Timestamp(d): g for d, g in predictions.groupby('date')}
    first = dates.get_indexer([predictions.date.min()])[0]
    last_signal = dates.get_indexer([predictions.date.max()])[0]
    start = first + 1
    stop = min(len(dates) - 1, last_signal + config['horizon'] + 1)
    shares = pd.Series(0.0, index=codes)
    cash, previous = 1.0, 1.0
    fee = cost_bps / 10000
    records = []
    for i in range(start, stop + 1):
        date = dates[i]
        prices = marked.loc[date].fillna(0)
        equity = float(cash + (shares * prices).sum())
        traded = 0.0
        blocked = 0
        stale = int(((shares > 0) & raw.loc[date].isna()).sum())
        signal_date = dates[i - 1]
        rebalance = ((i - 1 - first - offset) % config['horizon'] == 0) and (i - 1 - first >= offset) and signal_date in signals
        final = i == stop
        if rebalance or final:
            target = pd.Series(0.0, index=codes)
            if not final:
                group = signals[signal_date]
                if column == 'EqualWeight':
                    selected = group.code.tolist()
                else:
                    selected = group.sort_values([column, 'code'], ascending=[False, True]).head(config['top_k']).code.tolist()
                if selected:
                    target.loc[selected] = equity / (len(selected) * (1 + fee))
            tradable = (status.loc[date] == 1) & raw.loc[date].notna() & gap.loc[date].notna()
            buy_allowed = tradable & (gap.loc[date] < .095)
            sell_allowed = tradable & (gap.loc[date] > -.095)
            current = shares * prices
            sells = (current - target).clip(lower=0)
            blocked += int(((sells > 1e-12) & ~sell_allowed).sum())
            sells = sells.where(sell_allowed, 0)
            shares -= sells.div(prices.replace(0, np.nan)).fillna(0)
            cash += float(sells.sum()) * (1 - fee)
            traded += float(sells.sum())
            buys = (target - shares * prices).clip(lower=0)
            blocked += int(((buys > 1e-12) & ~buy_allowed).sum())
            buys = buys.where(buy_allowed, 0)
            if buys.sum() > 0:
                buys *= min(1.0, max(0, cash) / (float(buys.sum()) * (1 + fee)))
            shares += buys.div(prices.replace(0, np.nan)).fillna(0)
            cash -= float(buys.sum()) * (1 + fee)
            traded += float(buys.sum())
        value = float(cash + (shares * prices).sum())
        records.append({'date': date, 'equity': value, 'return': value / previous - 1, 'two_way_turnover': traded / equity if equity else 0, 'cash_weight': cash / value, 'blocked_orders': blocked, 'stale_positions': stale})
        previous = value
    path = pd.DataFrame(records)
    r = path['return']
    peaks = np.maximum.accumulate(np.r_[1.0, path.equity.to_numpy()])[1:]
    metrics = {'total_return': float(path.equity.iloc[-1] - 1), 'cagr': float(path.equity.iloc[-1] ** (252 / len(path)) - 1), 'sharpe': float(np.sqrt(252) * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0, 'max_drawdown': float((path.equity / peaks - 1).min()), 'annual_two_way_turnover': float(path.two_way_turnover.sum() * 252 / len(path)), 'blocked_orders': int(path.blocked_orders.sum()), 'stale_position_days': int(path.stale_positions.sum()), 'terminal_invested_weight': float(1 - path.cash_weight.iloc[-1]), 'n_days': len(path)}
    return path, metrics


def run(config: dict, manifest_path: Path, root: Path) -> Path:
    panel, dates, data_manifest = load_snapshot(manifest_path)
    frame, features = prepare(panel, dates, config)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_simple_baselines'
    destination = root / 'runs' / run_id
    destination.mkdir(parents=True)
    write_json(destination / 'config.json', config)
    write_json(destination / 'data_manifest.json', data_manifest)
    source_files = [p for folder in ('src', 'scripts/research', 'configs') for p in (root / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.json')]
    environment = {'run_id': run_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()), 'python': sys.version, 'platform': platform.platform(), 'command': [sys.executable, *sys.argv], 'packages': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost', 'baostock', 'matplotlib', 'pyarrow')}, 'source_sha256': {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in source_files}, 'data_manifest_sha256': sha256(manifest_path)}
    write_json(destination / 'environment.json', environment)
    folds, all_predictions, seed_records = [], [], []
    for year in config['test_years']:
        train, valid, test, fold = split_fold(frame, year, config['train_start'])
        prediction = test[['date', 'code', 'raw_return', 'excess_return', 'label_end']].copy()
        prediction['Momentum'] = test['return_20']
        fold['selection'] = {}
        for name, candidates in config['models'].items():
            start = time.perf_counter()
            scores, fitted = [], []
            for params in candidates:
                model = estimator(name, params, config['seeds'][0])
                model.fit(train[features], train.target)
                result = valid[['date', 'code', 'excess_return']].copy()
                result['prediction'] = model.predict(valid[features])
                score = daily_metrics(result, 'prediction', config['top_k'], config['minimum_stocks']).rank_ic.mean()
                if not np.isfinite(score):
                    raise ValueError(f'Invalid validation score for {name}')
                scores.append(float(score))
                fitted.append(model)
            best = int(np.argmax(scores))
            seed_predictions = []
            seeds = config['seeds'] if name in ('RandomForest', 'LightGBM', 'XGBoost') else config['seeds'][:1]
            for seed in seeds:
                model = fitted[best] if seed == seeds[0] else estimator(name, candidates[best], seed)
                if seed != seeds[0]:
                    model.fit(train[features], train.target)
                values = model.predict(test[features])
                seed_predictions.append(values)
                prediction[f'{name}_seed_{seed}'] = values
                result = test[['date', 'code', 'excess_return', 'label_end']].copy()
                result['prediction'] = values
                result = result.loc[result.label_end < pd.Timestamp(year + 1, 1, 1)]
                seed_records.append({'year': year, 'model': name, 'seed': seed, 'rank_ic': float(daily_metrics(result, 'prediction', config['top_k'], config['minimum_stocks']).rank_ic.mean())})
                joblib.dump(model, destination / f'{year}_{name}_{seed}.joblib')
            prediction[name] = np.mean(seed_predictions, axis=0)
            fold['selection'][name] = {'validation_rank_ic': scores, 'selected': candidates[best], 'seeds': seeds, 'seconds': round(time.perf_counter() - start, 3)}
            print(f'{year} {name}: validation={scores[best]:.4f}; {fold["selection"][name]["seconds"]}s', flush=True)
        folds.append(fold)
        all_predictions.append(prediction)
    predictions = pd.concat(all_predictions, ignore_index=True).sort_values(['date', 'code'])
    predictions.to_parquet(destination / 'predictions.parquet', index=False)
    names = ['Momentum', *config['models']]
    # All models share this exact outcome intersection. Never used for portfolio selection.
    evaluation = predictions.dropna(subset=names + ['excess_return']).copy()
    evaluation = evaluation.loc[evaluation.label_end.dt.year == evaluation.date.dt.year]
    counts = evaluation.groupby('date').size()
    evaluation = evaluation.loc[evaluation.date.isin(counts[counts >= config['minimum_stocks']].index)]
    daily, summaries = [], {}
    for name in names:
        d = daily_metrics(evaluation, name, config['top_k'], config['minimum_stocks'])
        d['model'] = name
        daily.append(d)
        summaries[name] = {'rank_ic': float(d.rank_ic.mean()), 'rank_ic_ci95': block_interval(d.rank_ic, config['bootstrap_block_dates'], config['bootstrap_repetitions']), 'ic': float(d.ic.mean()), 'rank_icir': float(d.rank_ic.mean() / d.rank_ic.std(ddof=1)), 'precision_at_k': float(d.precision.mean()), 'positive_excess_base_rate': float(d.base_rate.mean()), 'n_days': len(d)}
    daily = pd.concat(daily, ignore_index=True)
    daily.to_csv(destination / 'daily_metrics.csv', index=False)
    paired = daily.pivot(index='date', columns='model', values='rank_ic')
    differences = {name: {'mean': float((paired[name] - paired.Ridge).mean()), 'ci95': block_interval(paired[name] - paired.Ridge, config['bootstrap_block_dates'], config['bootstrap_repetitions'])} for name in names if name != 'Ridge'}
    portfolio_rows, paths, offsets = [], [], []
    for name in [*names, 'EqualWeight']:
        for cost in config['cost_bps']:
            path, metrics = portfolio(predictions, panel, dates, name, config, cost)
            portfolio_rows.append({'model': name, 'cost_bps': cost, **metrics})
            if cost == 10:
                path['model'] = name
                paths.append(path)
        for offset in range(config['horizon']):
            _, metrics = portfolio(predictions, panel, dates, name, config, 10, offset)
            offsets.append({'model': name, 'offset': offset, **metrics})
        print(f'Portfolio {name} complete', flush=True)
    portfolio_table = pd.DataFrame(portfolio_rows)
    portfolio_table.to_csv(destination / 'portfolio_metrics.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(destination / 'equity.csv', index=False)
    pd.DataFrame(offsets).to_csv(destination / 'offset_robustness.csv', index=False)
    pd.DataFrame(seed_records).to_csv(destination / 'seed_metrics.csv', index=False)
    coverage = {'download_rows': len(panel), 'fixed_universe_stocks': len(data_manifest['selected_codes']), 'eligible_prediction_rows': len(predictions), 'evaluated_rows': len(evaluation), 'prediction_dates': predictions.date.nunique(), 'evaluated_dates': evaluation.date.nunique(), 'missing_forward_outcome_rows': int(predictions.excess_return.isna().sum()), 'excluded_year_boundary_rows': int((predictions.label_end.dt.year != predictions.date.dt.year).sum()), 'min_stocks_per_evaluation_date': int(evaluation.groupby('date').size().min()), 'max_stocks_per_evaluation_date': int(evaluation.groupby('date').size().max()), 'per_stock': {code: {'rows': len(g), 'first': str(g.date.min().date()), 'last': str(g.date.max().date()), 'suspended_rows': int((g.tradestatus != 1).sum())} for code, g in panel.groupby('code')}}
    report = {'run_id': run_id, 'status': 'exploratory historical-cohort evidence; not a trading recommendation', 'config': config, 'features': features, 'coverage': coverage, 'folds': folds, 'predictive': summaries, 'paired_rank_ic_vs_ridge': differences, 'data_manifest': data_manifest, 'environment': environment}
    write_json(destination / 'report.json', report)
    from asharerank.evaluation.baseline_report import create_report
    create_report(destination, root / 'reports/baselines' / run_id)
    return destination
