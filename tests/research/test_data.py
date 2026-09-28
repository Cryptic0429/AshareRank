from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from asharerank.config import ExperimentConfig, MARKET_FEATURES, STOCK_FEATURES
from asharerank.data import build_common_dataset, build_common_features
from asharerank.schemas import CompareModelsRequest
from tests.research.helpers import synthetic_panel


class CommonDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()

    def test_common_feature_schema(self):
        features = build_common_features(self.panel)
        expected = {"date", "ts_code", *STOCK_FEATURES, *MARKET_FEATURES}
        self.assertEqual(expected, set(features.columns))
        self.assertFalse(features.duplicated(["date", "ts_code"]).any())

    def test_labels_and_chronological_split(self):
        dataset = build_common_dataset(self.panel, config=ExperimentConfig())
        self.assertTrue(dataset.frame["label"].between(-1, 1).all())
        self.assertLess(max(dataset.split.train_dates), min(dataset.split.valid_dates))
        self.assertLess(max(dataset.split.valid_dates), min(dataset.split.test_dates))
        self.assertFalse(
            set(dataset.split.train_dates) & set(dataset.split.test_dates)
        )

    def test_features_do_not_look_forward(self):
        cutoff = self.panel["date"].drop_duplicates().sort_values().iloc[60]
        changed = self.panel.copy()
        changed.loc[changed["date"] > cutoff, "close"] *= 5.0
        before = build_common_features(self.panel)
        after = build_common_features(changed)
        columns = [*STOCK_FEATURES, *MARKET_FEATURES]
        left = before.loc[before["date"] <= cutoff, columns].to_numpy(dtype=float)
        right = after.loc[after["date"] <= cutoff, columns].to_numpy(dtype=float)
        np.testing.assert_allclose(left, right, equal_nan=True)

    def test_comparison_request_defaults_to_fifty_stocks(self):
        self.assertEqual(CompareModelsRequest().max_stocks, 50)
        with self.assertRaises(ValidationError):
            CompareModelsRequest(max_stocks=51)


if __name__ == "__main__":
    unittest.main()
