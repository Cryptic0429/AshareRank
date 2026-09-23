from __future__ import annotations

import unittest

import pandas as pd

from asharerank.prediction.selection import balanced_top_k, balanced_universe_codes


class BalancedSelectionTests(unittest.TestCase):
    def test_balanced_top_k_round_robins_industries_without_changing_rank(self):
        frame = pd.DataFrame({
            "ts_code": [f"S{i}.SZ" for i in range(1, 9)],
            "industry": ["银行"] * 4 + ["电子"] * 2 + ["医药"] * 2,
            "raw_score": list(range(8, 0, -1)),
            "rank": list(range(1, 9)),
        })

        selected, method = balanced_top_k(frame, 4)

        self.assertEqual(method, "industry")
        self.assertEqual(selected["rank"].tolist(), [1, 5, 7, 2])
        self.assertEqual(selected["balanced_pick_rank"].tolist(), [1, 2, 3, 4])
        self.assertEqual(selected["rank"].nunique(), 4)

    def test_balanced_universe_uses_same_layer_for_all_models(self):
        codes = [f"S{i}.SZ" for i in range(1, 7)]
        info = pd.DataFrame({
            "ts_code": codes,
            "industry": ["银行", "银行", "银行", "电子", "电子", "医药"],
            "market_cap": [100, 90, 80, 70, 60, 50],
        })

        selected, distribution = balanced_universe_codes(codes, info, 4)

        self.assertEqual(len(selected), 4)
        self.assertEqual(set(distribution), {"银行", "电子", "医药"})
        self.assertLessEqual(max(distribution.values()) - min(distribution.values()), 1)


if __name__ == "__main__":
    unittest.main()
