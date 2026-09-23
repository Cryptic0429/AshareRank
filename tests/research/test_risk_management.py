from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from asharerank.portfolio import compute_account_state
from asharerank.risk.metrics import (
    assess_portfolio_risk,
    max_drawdown,
    summary_metrics,
    var_cvar_historical,
    var_cvar_weighted_historical,
)


class RiskMetricTests(unittest.TestCase):
    def test_max_drawdown_includes_loss_from_initial_capital(self):
        returns = pd.Series([-0.10, 0.02, 0.03])
        drawdown, series = max_drawdown(returns)
        self.assertAlmostEqual(drawdown, 0.10)
        self.assertAlmostEqual(series.iloc[0], 0.10)

    def test_weighted_history_reacts_to_clustered_recent_shocks(self):
        returns = pd.Series([0.0] * 98 + [-0.08, -0.10])
        historical_var, _ = var_cvar_historical(returns, 0.95)
        weighted_var, weighted_cvar = var_cvar_weighted_historical(returns, 0.95)
        self.assertAlmostEqual(historical_var, 0.0)
        self.assertGreater(weighted_var, historical_var)
        self.assertGreaterEqual(weighted_cvar, weighted_var)

    def test_cash_weight_reduces_portfolio_volatility(self):
        dates = pd.bdate_range("2025-01-01", periods=80)
        rng = np.random.default_rng(12)
        returns = pd.DataFrame({
            "A": rng.normal(0, 0.02, len(dates)),
            "B": rng.normal(0, 0.015, len(dates)),
        }, index=dates)
        cov = returns.cov()
        full = summary_metrics(returns, np.array([0.6, 0.4]), cov)
        half_cash = summary_metrics(returns, np.array([0.3, 0.2]), cov)
        self.assertAlmostEqual(
            half_cash["annualized_volatility"],
            full["annualized_volatility"] * 0.5,
        )

    def test_risk_assessment_explains_and_orders_components(self):
        result = assess_portfolio_risk(
            cvar95=0.055,
            annualized_volatility=0.30,
            max_drawdown_value=0.25,
            position_weights=np.array([0.55, 0.15, 0.10]),
            industry_weights=pd.Series({"Technology": 0.70, "Finance": 0.10}),
            stock_exposure=0.80,
        )
        self.assertIn(result["level"], {"high", "very_high"})
        self.assertEqual(len(result["top_drivers"]), 3)
        self.assertEqual(result["top_drivers"][0], "tail_loss")
        self.assertAlmostEqual(result["stock_exposure"], 0.8)

    def test_missing_cost_is_only_a_display_basis_for_unrealized_pnl(self):
        account = {
            "cash": 1000,
            "holdings": [{"ts_code": "600519.SH", "shares": 10, "cost_price": None}],
        }
        state = compute_account_state(account, pd.Series({"600519.SH": 100.0}))
        position = state["positions"].loc["600519.SH"]
        self.assertEqual(position["cost_price"], 100.0)
        self.assertEqual(position["unrealized_pnl"], 0.0)


if __name__ == "__main__":
    unittest.main()
