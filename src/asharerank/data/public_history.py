"""Immutable, checksum-verified BaoStock snapshots. No synthetic or provider fallback."""
from __future__ import annotations

import hashlib
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FIELDS = 'date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,isST'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def consume(result) -> pd.DataFrame:
    # Iteration avoids the removed pandas.DataFrame.append in SDK get_data().
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock {result.error_code}: {result.error_msg}')
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock incomplete response: {result.error_msg}')
    return pd.DataFrame(rows, columns=result.fields)


def acquire(config: dict, root: Path) -> Path:
    import baostock as bs

    identity = {k: config[k] for k in ('universe_date', 'universe_size', 'universe_seed', 'download_start', 'download_end')}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    directory = root / 'data/raw/baostock' / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for item in manifest['files']:
            if sha256(directory / item['path']) != item['sha256']:
                raise ValueError(f"Snapshot checksum mismatch: {item['path']}")
        return manifest_path
    socket.setdefaulttimeout(40)
    result = bs.login()
    if result.error_code != '0':
        raise RuntimeError(result.error_msg)
    try:
        snapshot = directory / 'universe.csv'
        if snapshot.exists():
            universe = pd.read_csv(snapshot, dtype=str)
        else:
            universe = consume(bs.query_hs300_stocks(date=config['universe_date']))
            if len(universe) != 300:
                raise ValueError(f'Expected 300 historical constituents, got {len(universe)}')
            if pd.to_datetime(universe.updateDate).max() > pd.Timestamp(config['universe_date']):
                raise ValueError('Provider returned a universe dated after requested as-of date')
            universe.to_csv(snapshot, index=False, encoding='utf-8')
        codes = sorted(universe.code, key=lambda c: hashlib.sha256((config['universe_seed'] + ':' + c).encode()).hexdigest())[:config['universe_size']]
        calendar_file = directory / 'calendar.csv'
        if not calendar_file.exists():
            consume(bs.query_trade_dates(config['download_start'], config['download_end'])).to_csv(calendar_file, index=False)
        files, failures = [], []
        for i, code in enumerate(codes, 1):
            target = directory / f'{code}.parquet'
            metadata_path = target.with_suffix('.json')
            if target.exists() and metadata_path.exists():
                info = json.loads(metadata_path.read_text(encoding='utf-8'))
                if sha256(target) != info['sha256']:
                    raise ValueError(f'Checksum mismatch for {code}')
                files.append(info)
                continue
            for attempt in range(3):
                try:
                    frame = consume(bs.query_history_k_data_plus(code, FIELDS, start_date=config['download_start'], end_date=config['download_end'], frequency='d', adjustflag='1'))
                    if frame.empty or frame.date.duplicated().any():
                        raise ValueError('Empty history or duplicate dates')
                    frame['date'] = pd.to_datetime(frame.date)
                    for column in set(frame) - {'date', 'code'}:
                        frame[column] = pd.to_numeric(frame[column], errors='coerce')
                    frame.to_parquet(target, index=False)
                    info = {'path': target.name, 'code': code, 'provider': 'baostock', 'api': 'query_history_k_data_plus', 'data_kind': 'real', 'adjustment': 'hfq / adjustflag=1', 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'requested_start': config['download_start'], 'requested_end': config['download_end'], 'rows': len(frame), 'start': str(frame.date.min().date()), 'end': str(frame.date.max().date()), 'missing_values': frame.isna().sum().to_dict(), 'sha256': sha256(target)}
                    write_json(metadata_path, info)
                    files.append(info)
                    print(f'[{i}/{len(codes)}] {code}: {len(frame)} rows', flush=True)
                    break
                except Exception as exc:
                    if attempt == 2:
                        failures.append({'code': code, 'error': str(exc)})
                        print(f'FAILED {code}: {exc}', flush=True)
                    else:
                        time.sleep(1 + attempt)
            time.sleep(0.15)
        if failures:
            write_json(directory / 'failures.json', failures)
            raise RuntimeError('Incomplete fixed universe: retry acquisition; stocks were not substituted')
        for name in ('universe.csv', 'calendar.csv'):
            files.append({'path': name, 'sha256': sha256(directory / name), 'provider': 'baostock', 'data_kind': 'real'})
        manifest = {'schema_version': 1, 'identity': identity, 'provider': 'baostock', 'data_kind': 'real', 'synthetic_rows': 0, 'fallback_used': False, 'universe_rule': 'Fixed historical CSI300 cohort; SHA256 ordering, no replacement based on outcomes', 'universe_requested_asof': config['universe_date'], 'universe_source_dates': sorted(universe.updateDate.unique().tolist()), 'selected_codes': codes, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'files': files, 'limitations': ['Historical cohort is not a dynamically reconstituted index or the full A-share market.', 'Provider historical revisions and adjustment methodology are not independently audited.', 'Delisting terminal proceeds and actual executable prices are not available.']}
        write_json(manifest_path, manifest)
        return manifest_path
    finally:
        bs.logout()


def load_snapshot(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    frames = []
    for item in manifest['files']:
        path = manifest_path.parent / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError(f'Changed snapshot: {path.name}')
        if path.suffix == '.parquet':
            frames.append(pd.read_parquet(path))
    calendar = pd.read_csv(manifest_path.parent / 'calendar.csv')
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_trading_day == 1, 'calendar_date']))
    return pd.concat(frames, ignore_index=True), dates, manifest
