"""全局配置：tushare token、缓存目录、默认参数。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = PROJECT_ROOT / "runs"
MODEL_DIR = RUNS_DIR / "models"
UNIVERSE_DIR = PROCESSED_DATA_DIR / "universe"

for _directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RUNS_DIR, MODEL_DIR, UNIVERSE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

TUSHARE_TOKEN: str = os.environ.get("TUSHARE_TOKEN", "")

# 风险计算默认参数
LOOKBACK_DAYS = 500          # 历史收益率回看窗口（交易日）
EWMA_LAMBDA = 0.94           # RiskMetrics 衰减因子
TRADING_DAYS_PER_YEAR = 252
MC_N_SIMULATIONS = 20000     # 蒙特卡洛模拟次数

# 预测模块
PREDICT_START_DATE = "20180101"   # 训练用行情起点
PREDICT_HORIZON = 5               # 前瞻收益天数
FEATURE_WARMUP = 60               # 特征滚动窗口预热天数
LGBM_TRAIN_YEARS = 2
LGBM_VALID_MONTHS = 3
LGBM_STEP_MONTHS = 3
