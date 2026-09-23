"""模型注册表：元数据 JSON + joblib 权重文件。"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib

from asharerank.settings import MODEL_DIR
from asharerank.predict.booster import _ensure_libomp

logger = logging.getLogger(__name__)
REGISTRY_FILE = MODEL_DIR / "registry.json"


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {"models": []}


def _save_registry(reg: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def list_models() -> list[dict]:
    return _load_registry().get("models", [])


def get_model_meta(model_id: str) -> dict | None:
    for m in list_models():
        if m["model_id"] == model_id:
            return m
    return None


def get_active_model_id() -> str | None:
    models = list_models()
    deployed = [m for m in models if m.get("status") == "已部署"]
    if deployed:
        return deployed[-1]["model_id"]
    return models[-1]["model_id"] if models else None


def set_deployed(model_id: str) -> dict:
    """把指定模型切换为线上激活模型（"已部署"），其余"已部署"模型自动降级为"历史"。

    与 `save_model()` / `save_stockmixer_model()` 里"新模型默认已部署"的逻辑相比，
    这是一个通用的、不关心 backend 类型的状态切换：不重新训练、不改权重文件，只是
    在已有的候选/历史模型之间切换"哪一个在线上生效"——用于「上线指标最好的那个候选」
    这类场景（见 `scripts/research/promote_best_model.py`）。

    刻意拒绝 `servable=False` 的实验性模型（只登记了指标，没有可反序列化的权重，
    `load_booster`/`stockmixer_infer` 都无法真正加载），避免把一个"看起来指标最好但
    根本没法推理"的模型错误地设为线上激活模型。
    """
    reg = _load_registry()
    models = reg.get("models", [])
    target = next((m for m in models if m["model_id"] == model_id), None)
    if target is None:
        raise ValueError(f"未知模型: {model_id}")
    if target.get("servable") is False:
        raise ValueError(
            f"模型 {model_id} 是实验性模型（只登记指标，没有可加载的权重文件），"
            f"不能直接设为线上激活模型；请先用对应的 deploy_*.py 脚本重新训练并落盘权重。"
        )
    for m in models:
        if m.get("status") == "已部署":
            m["status"] = "历史"
    target["status"] = "已部署"
    _save_registry(reg)
    logger.info("模型已切换为线上激活: %s", model_id)
    prune_family(_family_key(target))
    return target


def _family_key(m: dict) -> str:
    """模型家族键：LightGBM 和 HistGBM（缺 libomp 时的自动回退）算同一家族"lgbm"，
    因为二者本质是同一个"生产梯度提升模型"，没必要同时存在多份版本。其余按
    `backend` 字段分组（如 "stockmixer"）。"""
    backend = (m.get("backend") or "").lower()
    if backend in ("lightgbm", "sklearn_hgb"):
        return "lgbm"
    return backend or (m.get("type") or "unknown").lower()


def prune_family(family: str, metric: str = "rank_ic") -> list[str]:
    """同一模型家族只保留一个最优版本，删除其余的（registry 记录 + 权重文件）。

    排序优先级：当前是否"已部署" > `metric`（越大越好） > 训练时间（越新越好）。
    只处理 `servable != False` 的模型——实验性对照记录（见
    `register_experimental_model`）不参与，也不会被删除。

    由 `set_deployed()` / `save_model()` / `save_stockmixer_model()` 在新模型
    变为"已部署"时自动调用，保证"上线的模型只要一个最好版本"这条策略是持续生效
    的，不需要每次手动跑 `scripts/research/prune_models.py`。
    """
    members = [m for m in list_models() if m.get("servable") is not False and _family_key(m) == family]
    if len(members) <= 1:
        return []

    def sort_key(m: dict):
        deployed = 1 if m.get("status") == "已部署" else 0
        val = m.get(metric)
        val = val if isinstance(val, (int, float)) else float("-inf")
        return (deployed, val, m.get("trained_at") or "")

    ranked = sorted(members, key=sort_key, reverse=True)
    dropped = [m["model_id"] for m in ranked[1:]]
    for model_id in dropped:
        delete_model(model_id)
    if dropped:
        logger.info("模型家族 %s 已自动清理，只保留 %s: 删除 %s", family, ranked[0]["model_id"], dropped)
    return dropped


def delete_model(model_id: str) -> dict:
    """从 registry 移除一个模型记录，并删除它的权重文件（如果有）。

    用于清理同一模型家族里被淘汰的重复/历史候选版本（见
    `scripts/research/prune_models.py`），不影响其余模型的 status。只删除落在
    `MODEL_DIR` 下的权重文件（joblib/pt），避免路径异常时误删别处文件。
    """
    reg = _load_registry()
    models = reg.get("models", [])
    target = next((m for m in models if m["model_id"] == model_id), None)
    if target is None:
        raise ValueError(f"未知模型: {model_id}")
    reg["models"] = [m for m in models if m["model_id"] != model_id]
    _save_registry(reg)
    path = target.get("path")
    if path:
        p = Path(path)
        if p.exists() and p.resolve().parent == MODEL_DIR.resolve():
            p.unlink()
            logger.info("已删除权重文件: %s", p)
    logger.info("模型已从 registry 移除: %s", model_id)
    return target


def save_model(model_id: str, booster, feature_names: list[str],
               metrics: dict, extra: dict | None = None,
               status: str = "已部署") -> dict:
    """注册模型到 registry。

    Args:
        status: 默认"已部署"（生产训练流程 `train.py::run_training` 的行为不变：
            新模型自动成为线上激活模型，其余"已部署"模型自动降级为"历史"）。
            传 `status="候选"` 可以注册一个仅供对比评估、**不影响线上激活模型**
            的版本（例如训练数据/股票池对齐实验用的候选模型）——这种情况下不会
            触碰其他模型的 status，`get_active_model_id()` 也不会选中它。
    """
    path = MODEL_DIR / f"{model_id}.joblib"
    joblib.dump({"booster": booster, "feature_names": feature_names}, path)
    meta = {
        "model_id": model_id,
        "type": "LightGBM" if booster.backend == "lightgbm" else "HistGBM",
        "backend": booster.backend,
        "path": str(path),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": status,
        "ic": metrics.get("ic_mean"),
        "rank_ic": metrics.get("rank_ic_mean"),
        "icir": metrics.get("icir"),
        "rank_icir": metrics.get("rank_icir"),
        "annual_return": metrics.get("annual_return"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "n_features": len(feature_names),
        **(extra or {}),
    }
    reg = _load_registry()
    # 同 id 覆盖；只有新模型本身是"已部署"时，才把其余"已部署"模型降级为历史
    models = [m for m in reg["models"] if m["model_id"] != model_id]
    if status == "已部署":
        for m in models:
            if m.get("status") == "已部署":
                m["status"] = "历史"
    models.append(meta)
    reg["models"] = models
    _save_registry(reg)
    logger.info("模型已注册: %s（status=%s）IC=%.4f RankIC=%.4f", model_id, status,
                meta.get("ic") or 0, meta.get("rank_ic") or 0)
    if status == "已部署":
        prune_family(_family_key(meta))
    return meta


def save_stockmixer_model(model_id: str, ckpt_path, feature_names: list[str],
                          metrics: dict, extra: dict | None = None,
                          status: str = "已部署") -> dict:
    """注册一个可在线推理的 StockMixer 模型。

    与 `save_model()`（LightGBM/HistGBM，joblib 序列化）的区别：StockMixer 权重是
    `.pt`（torch state_dict + codes + feature_names + cfg + mean/std），不走 joblib，
    加载走 `stockmixer_infer.load_stockmixer_model`。显式设置 `servable=True`，
    因此**不会**被 `load_booster()` 的实验性模型检查拦下——它是真正接入 `/alpha`
    线上推理的模型，语义上和 `register_experimental_model()`（`servable=False`，
    只登记指标供模型实验室对比）完全不同。

    Args:
        ckpt_path: 已落盘在 `MODEL_DIR` 下的 `.pt` 权重文件路径（调用方负责把训练产出的
            checkpoint 复制/移动到这里，而不是直接指向 `models_experimental/` 下的临时文件，
            避免后续重新训练覆盖掉线上正在使用的权重）。
        metrics: 期望包含 `IC` / `RankIC` / `ICIR` / `Sharpe` / `AnnualizedReturn@K` 等
            `experimental/stockmixer/metrics.py::compute_metrics` 返回的键名。
    """
    path = Path(ckpt_path)
    meta = {
        "model_id": model_id,
        "type": "StockMixer",
        "backend": "stockmixer",
        "servable": True,
        "path": str(path),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": status,
        "ic": metrics.get("IC") if metrics.get("IC") is not None else metrics.get("ic_mean"),
        "rank_ic": metrics.get("RankIC") if metrics.get("RankIC") is not None else metrics.get("rank_ic_mean"),
        "icir": metrics.get("ICIR") if metrics.get("ICIR") is not None else metrics.get("icir"),
        "rank_icir": metrics.get("rank_icir"),
        "annual_return": next(
            (v for k, v in metrics.items() if k.startswith("AnnualizedReturn@")), metrics.get("annual_return")
        ),
        "sharpe": metrics.get("Sharpe") if metrics.get("Sharpe") is not None else metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "n_features": len(feature_names),
        **(extra or {}),
    }
    reg = _load_registry()
    models = [m for m in reg["models"] if m["model_id"] != model_id]
    if status == "已部署":
        for m in models:
            if m.get("status") == "已部署":
                m["status"] = "历史"
    models.append(meta)
    reg["models"] = models
    _save_registry(reg)
    logger.info("StockMixer 模型已注册（可在线推理）: %s（status=%s）IC=%.4f RankIC=%.4f", model_id, status,
               meta.get("ic") or 0, meta.get("rank_ic") or 0)
    if status == "已部署":
        prune_family(_family_key(meta))
    return meta


def load_booster(model_id: str | None = None):
    model_id = model_id or get_active_model_id()
    if not model_id:
        raise FileNotFoundError("尚无已训练模型，请先运行 scripts/research/train_lgbm.py")
    meta = get_model_meta(model_id)
    if meta is None:
        raise FileNotFoundError(f"未知模型: {model_id}")
    if meta.get("servable") is False:
        # 实验性模型（XGBoost / LSTM / StockMixer 等）：只登记指标供「模型实验室」
        # 页面对比展示，没有走 save_model() 落盘 joblib，也没有接入线上推理链路
        # （权重格式不同、特征管线不同）。真调用到这里说明有人手动传了这个 model_id
        # 去做在线推理——直接给出清晰报错，而不是让 joblib.load 因为找不到文件/
        # 反序列化失败报出一个让人摸不着头脑的异常。
        raise ValueError(
            f"模型 {model_id} 是实验性模型（{meta.get('type', '?')}"
            f"），暂未接入线上推理链路，仅供在模型实验室对比查看，不能用于 /alpha 推荐。"
        )
    # 反序列化 LightGBM Booster 前必须先注入 libomp 路径，
    # 否则在未曾调用过 get_backend() 的进程（如独立推理进程）里会 dlopen 失败。
    _ensure_libomp()
    blob = joblib.load(Path(meta["path"]))
    return blob["booster"], blob["feature_names"], meta


def register_experimental_model(model_id: str, *, model_type: str, metrics: dict,
                                extra: dict | None = None, status: str = "候选") -> dict:
    """登记一个「不走生产 joblib 序列化」的实验性模型（XGBoost / LSTM / StockMixer 等），
    只为了让它出现在「模型实验室」页面里和 lgbm 系模型一起对比查看，不落盘 booster、
    不接入 `load_booster` 的在线推理路径（`servable=False`，误调用会给出明确报错，
    见上面 `load_booster` 的检查）。

    和 `save_model()` 的区别：`save_model()` 要求传入一个真正的 booster 对象（要能
    `joblib.dump`、要有 `.backend` 属性），这里只登记指标 + 元信息，`path` 指向实验
    报告/权重文件的位置供人工查阅，不代表可以从这个路径反序列化出可预测的模型。
    """
    meta = {
        "model_id": model_id,
        "type": model_type,
        "backend": model_type.lower(),
        "servable": False,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": status,
        "ic": metrics.get("ic_mean") if metrics.get("ic_mean") is not None else metrics.get("ic"),
        "rank_ic": metrics.get("rank_ic_mean") if metrics.get("rank_ic_mean") is not None else metrics.get("rank_ic"),
        "icir": metrics.get("icir"),
        "rank_icir": metrics.get("rank_icir"),
        "annual_return": metrics.get("annual_return"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        **(extra or {}),
    }
    reg = _load_registry()
    models = [m for m in reg["models"] if m["model_id"] != model_id]
    if status == "已部署":
        for m in models:
            if m.get("status") == "已部署":
                m["status"] = "历史"
    models.append(meta)
    reg["models"] = models
    _save_registry(reg)
    logger.info("实验性模型已登记（不可在线推理，仅供对比）: %s（status=%s）RankIC=%.4f",
               model_id, status, meta.get("rank_ic") or 0)
    return meta
