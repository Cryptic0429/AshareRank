"""Scientific contracts for the new workflow. No external data calls."""
import numpy as np
import pandas as pd
import pytest

from asharerank.forecasting.simple_benchmark import block_interval, portfolio, prepare, split_fold


def test_boundary_uses_label_end_not_observed_row_count():
    dates = pd.bdate_range('2020-01-01', '2023-12-29')
    frame = pd.DataFrame({'date': dates, 'label_end': dates + pd.Timedelta(days=10), 'target': 0.1})
    train, valid, test, metadata = split_fold(frame, 2023, '2020-01-01')
    assert (train.label_end < pd.Timestamp('2022-01-01')).all()
    assert (valid.label_end < pd.Timestamp('2023-01-01')).all()
    assert metadata['train_excluded_dates']
    assert len(test) > 0


def flat_panel(dates):
    return pd.DataFrame({'date': dates, 'code': 'sh.600000', 'open': 10., 'close': 10., 'high': 10., 'low': 10., 'volume': 1e6, 'amount': 1e8, 'turn': 1., 'tradestatus': 1, 'isST': 0})


def test_entry_and_exit_costs_on_flat_prices():
    dates = pd.bdate_range('2023-01-02', periods=8)
    predictions = pd.DataFrame({'date': [dates[0]], 'code': ['sh.600000'], 'Ridge': [1.]})
    config = {'horizon': 5, 'top_k': 1}
    path, metrics = portfolio(predictions, flat_panel(dates), dates, 'Ridge', config, 10)
    assert path.equity.iloc[-1] == pytest.approx((1 - .001) / (1 + .001))
    assert metrics['max_drawdown'] < 0
    assert metrics['terminal_invested_weight'] == pytest.approx(0)


def test_missing_future_label_cannot_remove_portfolio_candidate():
    dates = pd.bdate_range('2023-01-02', periods=8)
    predictions = pd.DataFrame({'date': [dates[0]], 'code': ['sh.600000'], 'Ridge': [1.], 'excess_return': [np.nan]})
    path, _ = portfolio(predictions, flat_panel(dates), dates, 'Ridge', {'horizon': 5, 'top_k': 1}, 0)
    assert path.two_way_turnover.iloc[0] == pytest.approx(1)


def test_suspended_entry_leaves_cash():
    dates = pd.bdate_range('2023-01-02', periods=8)
    panel = flat_panel(dates)
    panel.loc[1, 'tradestatus'] = 0
    predictions = pd.DataFrame({'date': [dates[0]], 'code': ['sh.600000'], 'Ridge': [1.]})
    path, metrics = portfolio(predictions, panel, dates, 'Ridge', {'horizon': 5, 'top_k': 1}, 10)
    assert path.two_way_turnover.sum() == 0
    assert metrics['blocked_orders'] == 1


def test_features_do_not_use_future_prices_and_label_uses_next_open():
    dates = pd.bdate_range('2020-01-01', periods=100)
    panel = flat_panel(dates)
    panel['open'] = np.arange(100) + 10.
    panel['close'] = panel.open + .5
    config = {'horizon': 5, 'minimum_amount': 0}
    a, features = prepare(panel, dates, config)
    altered = panel.copy()
    altered.loc[altered.date > dates[80], ['open', 'close']] *= 2
    b, _ = prepare(altered, dates, config)
    pd.testing.assert_frame_equal(a.loc[a.date <= dates[80], features], b.loc[b.date <= dates[80], features])
    value = a.loc[a.date == dates[70], 'raw_return'].iloc[0]
    assert value == pytest.approx(panel.open.iloc[76] / panel.open.iloc[71] - 1)


def test_block_interval_constant_series():
    assert block_interval(np.ones(50)) == [1., 1.]
