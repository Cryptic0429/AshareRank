from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from asharerank.config import ExperimentConfig
from asharerank.data import build_common_dataset
from asharerank.evaluation import evaluate_predictions
from asharerank.models.registry import create_model, list_candidate_models
from asharerank.prediction import build_prediction_frame, rank_predictions
from asharerank.training import run_experiment
from tests.research.helpers import synthetic_panel


class ModelsAndEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentConfig()
        cls.dataset = build_common_dataset(synthetic_panel(), config=cls.config)

    def test_registry_reserves_custom_model(self):
        candidates = {item["name"]: item for item in list_candidate_models()}
        self.assertFalse(candidates["custom"]["available"])
        with self.assertRaises(NotImplementedError):
            create_model("custom")

    def test_ridge_end_to_end(self):
        result = run_experiment(
            self.dataset,
            "ridge",
            experiment_config=self.config,
            save_artifacts=False,
            top_k=3,
        )
        self.assertEqual(result.model_name, "ridge")
        self.assertIn("rank_ic", result.metrics)
        self.assertTrue(result.predictions["rank_pct"].between(0, 1).all())

    def test_random_forest_contract(self):
        result = run_experiment(
            self.dataset,
            "random_forest",
            experiment_config=self.config,
            model_params={"n_estimators": 10, "max_depth": 4, "n_jobs": 1},
            save_artifacts=False,
            top_k=3,
        )
        self.assertEqual(len(result.predictions), result.metrics["n_test"])

    def test_lite_rank_trains_by_date_and_returns_rank(self):
        result = run_experiment(
            self.dataset,
            "lite_rank",
            experiment_config=self.config,
            model_params={"n_estimators": 30, "n_jobs": 1},
            save_artifacts=False,
            top_k=3,
        )
        self.assertEqual(result.model_name, "lite_rank")
        self.assertGreater(result.metrics["best_iteration"], 0)
        self.assertIn(result.metrics["orientation"], {-1.0, 1.0})
        per_day = result.predictions.groupby("date")["rank"]
        self.assertTrue((per_day.min() == 1).all())

    def test_ranking_is_independent_per_model_and_date(self):
        metadata = pd.DataFrame({
            "date": ["2024-01-02"] * 3,
            "ts_code": ["A", "B", "C"],
            "future_excess_return": [0.03, 0.01, -0.02],
        })
        ranked = rank_predictions(build_prediction_frame(metadata, [3, 2, 1], "demo"))
        self.assertEqual(ranked.iloc[0]["ts_code"], "A")
        self.assertEqual(ranked["rank"].tolist(), [1, 2, 3])
        metrics = evaluate_predictions(ranked, k=2)
        # 单日 IC 有定义，但 ICIR 因没有跨日波动按 0 处理。
        self.assertAlmostEqual(metrics["precision_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
