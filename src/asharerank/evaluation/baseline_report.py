"""Curated bilingual reports and figures from saved baseline run artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asharerank.data.public_history import write_json

COLORS = ['#666666', '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
STYLES = ['--', '-', '-.', ':', '--', '-.', ':']


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def save(fig, output: Path, name: str):
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(output / f'{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_report(run: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    config, coverage = report['config'], report['coverage']
    daily = pd.read_csv(run / 'daily_metrics.csv', parse_dates=['date'])
    equity = pd.read_csv(run / 'equity.csv', parse_dates=['date'])
    portfolios = pd.read_csv(run / 'portfolio_metrics.csv')
    offsets = pd.read_csv(run / 'offset_robustness.csv')
    seeds = pd.read_csv(run / 'seed_metrics.csv')
    names = list(report['predictive'])
    all_names = [*names, 'EqualWeight']
    color = dict(zip(all_names, COLORS))
    style = dict(zip(all_names, STYLES))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 13, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for i, name in enumerate(names):
        m = report['predictive'][name]
        lower, upper = m['rank_ic_ci95']
        axes[0].errorbar(m['rank_ic'], i, xerr=[[max(0, m['rank_ic'] - lower)], [max(0, upper - m['rank_ic'])]], fmt='o', color=color[name], capsize=4, markersize=7)
    axes[0].set(yticks=range(len(names)), yticklabels=names, xlabel='Mean daily Spearman RankIC', title='A  Ranking quality with 95% intervals')
    axes[0].axvline(0, color='#bbbbbb', lw=1)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=.15)
    paired = report['paired_rank_ic_vs_ridge']
    for i, (name, m) in enumerate(paired.items()):
        lo, hi = m['ci95']
        axes[1].plot([lo, hi], [i, i], color=color[name], lw=2)
        axes[1].scatter([m['mean']], [i], color=color[name], s=45)
    axes[1].set(yticks=range(len(paired)), yticklabels=list(paired), xlabel='Paired mean RankIC difference vs Ridge', title='B  Does complexity improve on Ridge?')
    axes[1].axvline(0, color='#777777', lw=1, linestyle='--')
    axes[1].invert_yaxis()
    fig.suptitle('Historical CSI300 cohort | 2023–2025 out-of-sample comparison', fontsize=15)
    fig.supxlabel('Circular moving-block bootstrap: 20 dates, 1,000 resamples. Exploratory, no multiplicity adjustment.', fontsize=10)
    save(fig, output, 'ranking')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained', sharex=True, height_ratios=[2, 1])
    for name in all_names:
        d = equity.loc[equity.model == name]
        axes[0].plot(d.date, d.equity, color=color[name], linestyle=style[name], label=name, linewidth=1.7)
        peaks = np.maximum.accumulate(np.r_[1, d.equity])[1:]
        axes[1].plot(d.date, (d.equity / peaks - 1) * 100, color=color[name], linestyle=style[name], linewidth=1.3)
    axes[0].set(title='A  Open-price execution proxy, 10 bps per traded side', ylabel='Portfolio value (initial = 1)')
    axes[0].legend(ncol=4, fontsize=10, frameon=False)
    axes[1].set(title='B  Mark-to-market drawdown', ylabel='Drawdown (%)', xlabel='Date')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.supxlabel('5-session rebalance; blocked orders remain unfilled; missing held prices use stale marks. Not executable performance.', fontsize=9)
    save(fig, output, 'portfolio')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for name in all_names:
        d = portfolios.loc[portfolios.model == name]
        axes[0].plot(d.cost_bps, d.cagr * 100, marker='o', linestyle=style[name], color=color[name], label=name)
        d = offsets.loc[offsets.model == name]
        axes[1].plot(d.offset, d.cagr * 100, marker='o', linestyle=style[name], color=color[name])
    axes[0].set(title='A  Cost sensitivity', xlabel='Cost per traded side (bps)', ylabel='Annualized return (%)')
    axes[1].set(title='B  Rebalance-start sensitivity (10 bps)', xlabel='Signal offset (trading sessions)', ylabel='Annualized return (%)', xticks=range(5))
    for ax in axes:
        ax.axhline(0, color='#bbbbbb', lw=1)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9, frameon=False, ncol=2)
    save(fig, output, 'robustness')

    yearly = daily.assign(year=daily.date.dt.year).groupby(['model', 'year']).rank_ic.mean().unstack().reindex(names)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    maximum = max(.01, abs(yearly.to_numpy()).max())
    img = axes[0].imshow(yearly, cmap='RdBu', vmin=-maximum, vmax=maximum, aspect='auto')
    for i in range(len(names)):
        for j in range(len(yearly.columns)):
            axes[0].text(j, i, f'{yearly.iloc[i,j]:.3f}', ha='center', va='center', color='white' if abs(yearly.iloc[i,j]) > maximum * .55 else 'black')
    axes[0].set(xticks=range(len(yearly.columns)), xticklabels=yearly.columns, yticks=range(len(names)), yticklabels=names, title='A  RankIC by test year')
    fig.colorbar(img, ax=axes[0], label='Mean RankIC', shrink=.8)
    for i, name in enumerate(config['models']):
        d = seeds.loc[seeds.model == name]
        for j, year in enumerate(config['test_years']):
            values = d.loc[d.year == year, 'rank_ic'].to_numpy()
            axes[1].scatter(np.full(len(values), i) + (j - 1) * .2, values, marker=['o', 's', '^'][j], color=color[name], alpha=.8)
    axes[1].set(xticks=range(len(config['models'])), xticklabels=list(config['models']), ylabel='Individual-seed RankIC', title='B  Seed and period sensitivity')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].axhline(0, color='#bbbbbb', lw=1)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color='black', marker=m, linestyle='', label=str(y)) for m, y in zip(['o', 's', '^'], config['test_years'])], frameon=False)
    save(fig, output, 'stability')

    prediction_rows = []
    for name, m in report['predictive'].items():
        prediction_rows.append([name, f"{m['rank_ic']:.4f}", f"[{m['rank_ic_ci95'][0]:.4f}, {m['rank_ic_ci95'][1]:.4f}]", f"{m['ic']:.4f}", f"{m['rank_icir']:.3f}", f"{m['precision_at_k']:.1%}"])
    ptable = table(['Model', 'RankIC', '95% block CI', 'IC', 'RankICIR', 'Precision@10'], prediction_rows)
    net = portfolios.loc[portfolios.cost_bps == 10].set_index('model')
    ntable = table(['Model', 'CAGR', 'Sharpe', 'Max drawdown', 'Annual 2-way turnover', 'Blocked orders', 'Stale position-days'], [[n, f'{r.cagr:.1%}', f'{r.sharpe:.2f}', f'{r.max_drawdown:.1%}', f'{r.annual_two_way_turnover:.1f}x', int(r.blocked_orders), int(r.stale_position_days)] for n, r in net.iterrows()])
    best = max(names, key=lambda n: report['predictive'][n]['rank_ic'])
    improvements = [n for n, m in paired.items() if n != 'Momentum' and m['ci95'][0] > 0]
    limitations = [
        'Fixed 60-stock historical CSI300 cohort, not all A-shares or a dynamically reconstituted index; a single cohort size and five-day horizon were studied.',
        'Constituents were requested as of 2018-01-02; provider update dates and all selected codes are preserved. Historical vendor corrections remain possible.',
        'The same training-only median imputer and scaler workflow is fitted separately per model/fold. Daily feature ranks use only signal-date eligible stocks.',
        'Forward returns are next-session adjusted open to the adjusted open five sessions later. Excess returns subtract the equally weighted observed eligible-cohort return; this is not CSI300 index excess return.',
        'Final six signal dates per annual fold can lose forward-label coverage; these observations are excluded from predictive metrics, never from portfolio candidate selection on the basis of future returns.',
        'Backtests are an OHLC execution proxy. The conservative 9.5% open-gap screen is not an exact historical exchange price-limit model (including ChiNext changes). No order book, impact, participation cap, or lot-size model is included.',
        'Suspended or screened orders remain unfilled. Missing held prices are forward-filled for valuation only. Delisting/merger terminal proceeds are unknown: stale position-days and remaining invested terminal weight must be inspected; proxy returns may be biased.',
        'One cost scenario includes all friction as a constant per traded side; actual commissions, sell-side stamp tax, and spread vary over time. Sharpe uses zero risk-free rate and 252 sessions/year.',
        'Intervals use 20-date circular moving blocks and 1,000 resamples; paired comparisons are unadjusted exploratory intervals, not multiplicity-controlled hypothesis tests.',
        'Tree model headlines average three predetermined seeds; linear models are deterministic. Hyperparameters are chosen on the preceding validation year only. Prior test years become historical training/validation data in later folds.',
        'StockMixer and LSTM were deliberately excluded. No automated test suite was run because project instructions require an explicit testing request.'
    ]
    run_id = report['run_id']
    common = f"Run ID: `{run_id}`  \nSource revision: `{report['environment']['git_revision']}` (dirty={report['environment']['git_dirty']})  \nReproduce: `python scripts/research/run_public_baselines.py --config configs/public_baselines.json`\n"
    references = '\n- [BaoStock data provider](https://www.baostock.com/)\n- [BaoStock adjustment-factor methodology](https://www.baostock.com/helpdocs/pdf/BaoStock%E5%A4%8D%E6%9D%83%E5%9B%A0%E5%AD%90%E7%AE%80%E4%BB%8B.pdf)\n- [AKShare alternative source documentation](https://github.com/akfamily/akshare/blob/main/docs/data/stock/stock.md) — explored, not used in this run.\n'
    en = f"""# Simple-model comparison: historical A-share cohort

{common}
## Findings

The highest observed mean RankIC is **{best}** ({report['predictive'][best]['rank_ic']:.4f}). Models with a positive unadjusted paired 95% interval versus Ridge: **{', '.join(improvements) or 'none'}**. This is an exploratory comparison, not proof of general superiority or profitable deployability.

The run contains {coverage['download_rows']:,} real source rows from {coverage['fixed_universe_stocks']} fixed historical constituents. The common predictive sample has {coverage['evaluated_rows']:,} observations over {coverage['evaluated_dates']} dates, with {coverage['min_stocks_per_evaluation_date']}–{coverage['max_stocks_per_evaluation_date']} eligible stocks per date. {coverage['missing_forward_outcome_rows']:,} prediction rows lack forward outcomes. Portfolio selection retains the signal-time universe independently of label availability.

## Predictive comparison

{ptable}

![Ranking uncertainty and paired differences](ranking.png)

RankIC is daily Spearman correlation with five-session cohort-excess returns. RankICIR is the daily mean divided by daily sample standard deviation, not annualized. Precision@10 is the proportion of selected stocks with positive cohort-excess return; the observed baseline positive rate is {report['predictive'][best]['positive_excess_base_rate']:.1%}.

## Portfolio proxy after costs

{ntable}

![Portfolio value and drawdown](portfolio.png)

Top 10 equal weights, five-session rebalancing, next-open execution, 10 bps per traded side. EqualWeight uses all signal-eligible cohort stocks under the same execution and cost conventions. It is a cohort control, not the published CSI300 index. Sharpe and drawdown use daily open-price marks, including the initial capital of 1.

![Cost and start-offset sensitivity](robustness.png)

## Stability

![Year and seed comparison](stability.png)

{table(['Model', *map(str, yearly.columns)], [[n, *[f'{v:.4f}' for v in yearly.loc[n]]] for n in names])}

## Protocol and limitations

""" + '\n\n'.join(f'{i+1}. {s}' for i, s in enumerate(limitations)) + f"\n\n## Data and reproducibility\n\nCurated configuration, checksums, exclusions, coverage and aggregate metrics are in [summary.json](summary.json). Local raw snapshots, model weights and per-observation predictions remain in ignored `data/` and `runs/{run_id}/`. Use the archived snapshot to reproduce exact values; re-downloading vendor data may yield revisions.\n\n## Sources\n{references}"
    zh = f"""# 简单模型比较：A 股历史股票池

{common}
## 主要发现

本次样本外平均 RankIC 最高的是 **{best}**（{report['predictive'][best]['rank_ic']:.4f}）。与 Ridge 的配对差值之未经多重比较校正的 95% 区间完全高于零的模型：**{', '.join(improvements) or '无'}**。这是探索性结果，不能据此断言模型普遍更优或策略可以实盘盈利。

实际获取 **{coverage['download_rows']:,} 行真实行情**，股票池为 2018 年初沪深 300 成分股中预先按 SHA256 顺序选出的 **{coverage['fixed_universe_stocks']} 只**，不根据未来表现替换股票。共同评价样本为 **{coverage['evaluated_rows']:,} 条、{coverage['evaluated_dates']} 个交易日**，每天覆盖 {coverage['min_stocks_per_evaluation_date']}–{coverage['max_stocks_per_evaluation_date']} 只股票。缺少未来收益的预测记录为 {coverage['missing_forward_outcome_rows']:,} 条；组合选股不以未来标签是否可用来筛选候选。

## 排序指标与不确定性

{ptable}

![排序指标及相对 Ridge 的配对差值](ranking.png)

左图为平均 RankIC 和区块重采样区间，右图为相对 Ridge 的同日配对差值。区间跨越零时，当前数据不足以明确区分。RankIC 是逐日 Spearman 秩相关；RankICIR 为逐日均值除以样本标准差，不年化。Precision@10 为前十名中未来股票池超额收益为正的比例，其全体样本正收益基准比例为 {report['predictive'][best]['positive_excess_base_rate']:.1%}。

## 扣除成本后的组合代理评价

{ntable}

![组合净值与回撤](portfolio.png)

采用 Top 10 等权、每五个交易日再平衡、信号后下一交易日开盘执行，表中成本为每成交方向 10 bps。基准 EqualWeight 持有当时所有符合条件的股票，使用相同执行和费用规则，不是官方沪深 300 指数。年化收益为 CAGR；夏普使用日收益、零无风险利率和每年 252 个交易日；换手按买入与卖出金额之和计算。

停牌或开盘涨跌幅超过保守阈值的订单不成交。**这仍是基于日线的成交代理，不是可实盘复制的收益**。尤其需检查表中的 Stale position-days（缺失持仓价格而沿用旧估值的天数）以及 summary.json 中的期末未变现仓位；退市、合并的最终现金流并未被完整识别。

![成本与再平衡起始日敏感性](robustness.png)

## 跨年份与随机种子稳定性

![按年份与随机种子的排序指标](stability.png)

{table(['模型', *map(str, yearly.columns)], [[n, *[f'{v:.4f}' for v in yearly.loc[n]]] for n in names])}

树模型主结果为预先固定的三个种子等权平均；线性模型为确定性拟合。每个模型只在上一年验证集上选择两组候选参数之一，所有预处理仅在训练集拟合。2023、2024、2025 年依次作为样本外测试，历史测试年份随后可以作为后续折的历史训练或验证资料。StockMixer 与 LSTM 本轮不运行。

## 数据口径与限制

- 采用 BaoStock 后复权日线，特征仅用信号日及以前的数据；标签为下一交易日开盘到再往后五个交易日开盘的收益，扣除当日可观测股票池等权前瞻收益。该超额收益不等同于相对官方指数的超额收益。
- 标签先构建再切分。只保留标签终点严格早于后续分区的训练/验证样本；由于延后一天执行，清除范围覆盖五日持有期加一天执行延迟。逐折实际排除日期保存在 summary.json。
- 股票池为固定历史队列，不代表全 A 股或动态指数成分；本轮只研究一个股票池规模与五日标签期限，不能宣称已完成所有稳健性检验。
- 行情来源、下载时刻、校验和与缺失信息均记录；没有使用合成数据或静默回退。历史数据仍可能被供应商修订，复权与最终退市结算没有独立审计。
- 开盘 ±9.5% 的保守拦截不是精确的历史涨跌停规则（例如创业板制度变化）。未模拟盘口、冲击、参与率上限和整手约束；真实税费、佣金与价差也会随时间变化。
- 区间来自 20 日循环移动区块、1,000 次重采样，保留一定序列相关性；未进行多重检验校正，不是预注册的确认性结论。
- 遵循仓库要求，本次没有运行自动化测试套件；新增科学转换的测试文件已提供，需明确要求后执行。

## 复现与来源

参见 [summary.json](summary.json) 中的配置、来源清单、分区和摘要。原始数据、逐样本预测与模型保留在本地忽略目录 `data/` 和 `runs/{run_id}/`，不上传 GitHub。精确复现须使用同一校验和的数据快照，重新下载可能得到修订版本。

{references}
"""
    (output / 'README.md').write_text(zh, encoding='utf-8')
    (output / 'README.en.md').write_text(en, encoding='utf-8')
    environment = {k: report['environment'][k] for k in ('git_revision', 'git_dirty', 'packages', 'source_sha256', 'data_manifest_sha256')}
    summary = {k: report[k] for k in ('run_id', 'status', 'config', 'coverage', 'features', 'folds', 'predictive', 'paired_rank_ic_vs_ridge', 'data_manifest')}
    summary.update(environment=environment, portfolio_metrics=portfolios.to_dict('records'), offset_robustness=offsets.to_dict('records'), seed_metrics=seeds.to_dict('records'), limitations=limitations)
    write_json(output / 'summary.json', summary)
    (output.parent / 'README.md').write_text(f'# Baseline research reports\n\n- [{run_id} — 中文]({run_id}/README.md) · [English]({run_id}/README.en.md)\n\nExploratory historical-cohort analysis. Read coverage and execution limitations alongside all metrics.\n', encoding='utf-8')
