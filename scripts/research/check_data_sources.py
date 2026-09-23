"""校验当前持仓 + 沪深300 指数行情是否为真实数据源（tushare/akshare），
而非静默降级为模拟数据（synthetic）。建议在正式使用/演示风险管理前先跑一遍，
以及每次修改持仓（新增/替换标的）后重新跑一遍。

用法：
    python scripts/research/check_data_sources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from asharerank import data_service as ds  # noqa: E402
from asharerank.portfolio import load_account  # noqa: E402


def main() -> int:
    account = load_account()
    codes = [h["ts_code"] for h in account["holdings"]]
    print(f"持仓标的：{len(codes)} 只\n")

    prices = ds.get_price_history(codes)
    source_map = ds.get_price_source_map(codes)

    ok = True
    for code in codes:
        src = source_map.get(code, "synthetic")
        real = src in ("tushare", "akshare")
        ok = ok and real
        mark = "✅" if real else "❌"
        last_price = prices[code].iloc[-1] if code in prices.columns else float("nan")
        print(f"  {mark} {code:<12} 数据源={src:<10} 最新收盘价={last_price:.2f}")

    idx = ds.get_index_history(full=False)
    print(f"\n沪深300 指数：{len(idx)} 行，最新交易日 {idx.index[-1].date()}")
    print(f"整体数据源判定（取最劣）：{ds.data_source_status()}")

    if not ok:
        print("\n⚠️ 存在标的降级为模拟数据，相关风险指标（VaR/因子/压力测试）可信度会下降。")
        print("排查建议：")
        print("  1) .env 中 TUSHARE_TOKEN 是否有效、额度是否充足；")
        print("  2) 当前网络是否可访问 tushare/akshare 接口；")
        print("  3) 股票代码格式是否正确（如 600519.SH / 000858.SZ / 300750.SZ）；")
        print("  4) data/ 下对应 daily_*.csv 是否过旧（超过 3 天会重新拉取）。")
        return 1

    print("\n✅ 全部标的均使用真实行情数据，可放心用于风险计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
