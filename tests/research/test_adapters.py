from __future__ import annotations

import unittest

from asharerank.adapters import MasterAdapter, TabularAdapter
from asharerank.config import ExperimentConfig
from asharerank.data import build_common_dataset
from tests.panel.helpers import synthetic_panel


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentConfig()
        cls.dataset = build_common_dataset(synthetic_panel(), config=cls.config)
        cls.train = cls.dataset.partition("train")

    def test_tabular_shape(self):
        adapter = TabularAdapter(lookback=8).fit(self.train)
        data = adapter.transform(self.dataset.frame)
        self.assertEqual(data.X.shape[1], 8 * 26)
        self.assertEqual(len(data.X), len(data.y))

    def test_master_uses_same_samples(self):
        tabular = TabularAdapter(lookback=8).fit(self.train).transform(self.dataset.frame)
        master = MasterAdapter(lookback=8).fit(self.train).transform(self.dataset.frame)
        self.assertEqual(tabular.metadata[["date", "ts_code"]].to_dict("records"),
                         master.metadata[["date", "ts_code"]].to_dict("records"))
        self.assertEqual(master.X["stock"].shape[1:], (8, 18))
        self.assertEqual(master.X["market"].shape[1:], (8, 8))


if __name__ == "__main__":
    unittest.main()
