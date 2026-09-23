"""SQLite 持久化：风险指标时间序列、告警记录、预测结果（预留）。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from asharerank.settings import RUNS_DIR

DB_FILE = RUNS_DIR / "metadata" / "risk.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_metrics_daily (
    date        TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL,
    created_at  TEXT DEFAULT (datetime('now', 'localtime')),
    PRIMARY KEY (date, metric)
);
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    level       TEXT NOT NULL,
    rule        TEXT NOT NULL,
    message     TEXT NOT NULL,
    status      TEXT DEFAULT 'open',
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS predictions (
    date        TEXT NOT NULL,
    ts_code     TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    alpha_score REAL,
    rank_pct    REAL,
    PRIMARY KEY (date, ts_code, model_id)
);
"""


@contextmanager
def get_conn():
    DB_FILE.parent.mkdir(exist_ok=True, parents=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


def save_risk_metrics(date: str, metrics: dict[str, float]) -> None:
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO risk_metrics_daily (date, metric, value) VALUES (?, ?, ?)",
            [(date, k, float(v)) for k, v in metrics.items() if v is not None],
        )


def save_alerts(date: str, alerts: list[dict]) -> None:
    """同一日期的告警先清后写，避免重复计算产生重复记录。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM alerts WHERE date = ?", (date,))
        conn.executemany(
            "INSERT INTO alerts (date, level, rule, message) VALUES (?, ?, ?, ?)",
            [(date, a["level"], a["rule"], a["message"]) for a in alerts],
        )


def load_metric_history(metric: str, limit: int = 250) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT date, value FROM risk_metrics_daily WHERE metric = ? ORDER BY date DESC LIMIT ?",
            (metric, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def load_alert_history(limit: int = 100) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, date, level, rule, message, status, created_at "
            "FROM alerts ORDER BY date DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
