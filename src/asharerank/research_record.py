"""Common audit records for research commands, including failures and unknown lineage."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

_inputs: dict[str, dict] = {}
_arguments: dict = {}
_active = None


def record_arguments(arguments) -> None:
    global _arguments
    value = dict(vars(arguments)) if hasattr(arguments, '__dict__') else dict(arguments)
    _arguments = json.loads(json.dumps(value, default=str))


def register_input(path: Path, frame=None, *, role='market_data') -> None:
    path = Path(path).resolve()
    info = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'role': role, 'source_status': 'unknown', 'data_kind': 'unknown'}
    if frame is not None:
        info['rows'] = len(frame)
        if '_provider' in frame:
            info['providers'] = sorted(frame['_provider'].fillna('unknown').unique().tolist())
            info['unknown_source_rows'] = int(frame['_provider'].isna().sum() + (frame['_provider'] == 'unknown').sum())
            info['source_status'] = 'complete' if info['unknown_source_rows'] == 0 else 'partial'
        else:
            info['unknown_source_rows'] = len(frame)
        for field in ('_retrieved_at', '_adjustment', '_data_kind'):
            if field in frame:
                info[field.lstrip('_')] = sorted(frame[field].fillna('unknown').astype(str).unique().tolist())
        if '_data_kind' in frame and frame['_data_kind'].eq('real').all():
            info['data_kind'] = 'real'
        if 'date' in frame and len(frame):
            info['date_range'] = [str(frame.date.min()), str(frame.date.max())]
        info['missing_values'] = {str(k): int(v) for k, v in frame.isna().sum().items()}
    _inputs[str(path)] = info


def input_manifest():
    values = copy.deepcopy(list(_inputs.values()))
    return {'scope': 'Files read through instrumented loaders in this process; may be a superset for an individual model. Uninstrumented external inputs are not covered.', 'files': values, 'provenance_complete_for_registered_files': bool(values) and all(v['source_status'] == 'complete' for v in values), 'limitations': ['Legacy caches without explicit row provenance remain unknown; source is never inferred from a filename.', 'Registration alone does not prove point-in-time correctness or complete external data coverage.']}


def environment_record():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return 'unknown'
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'lightgbm', 'xgboost', 'torch', 'baostock', 'akshare', 'tushare'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    source_paths = [p for folder in ('src', 'scripts/research', 'configs') for p in (root / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.json')]
    return {'git_revision': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')), 'python': sys.version, 'platform': platform.platform(), 'packages': packages, 'command': [sys.executable, *sys.argv], 'resolved_cli_arguments': dict(_arguments), 'source_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}


def run_recorded(function):
    global _active, _inputs, _arguments
    if _active is not None or any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        return function()
    root = Path(__file__).resolve().parents[2]
    started = datetime.now(timezone.utc)
    run_id = started.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + Path(sys.argv[0]).stem
    destination = root / 'runs/commands' / run_id
    destination.mkdir(parents=True)
    _active, _inputs, _arguments = run_id, {}, {}
    record = {'schema_version': 1, 'run_id': run_id, 'started_at_utc': started.isoformat(), 'status': 'running', **environment_record()}
    def persist():
        (destination / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    persist()
    try:
        result = function()
        record['status'] = 'succeeded'
        return result
    except BaseException as exc:
        record.update(status='failed', exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), resolved_cli_arguments=dict(_arguments), data_manifest=input_manifest())
        persist()
        _active = None
