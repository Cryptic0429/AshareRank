from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from asharerank.adapters.base import AdaptedData
from asharerank.prediction.inference.engine import predict_saved_models, prepare_latest_feature_frame
from asharerank.schemas import LivePredictRequest
from tests.research.helpers import synthetic_panel


class _InferenceOnlyAdapter:
    def transform(self, frame):
        rows = frame[frame["label"].notna()][
            ["date", "ts_code", "volatility_20"]
        ].reset_index(drop=True)
        return AdaptedData(
            X=np.zeros((len(rows), 1), dtype=np.float32),
            y=np.zeros(len(rows), dtype=np.float32),
            metadata=rows,
            feature_names=("demo",),
        )


class _InferenceOnlyModel:
    estimator = None

    def fit(self, *args, **kwargs):
        raise AssertionError("在线推理不应调用 fit")

    def predict(self, X):
        return np.arange(len(X), dtype=float)


class LiveInferenceTests(unittest.TestCase):
    def test_latest_frame_uses_common_date_without_future_label(self):
        frame, prediction_date, coverage = prepare_latest_feature_frame(
            synthetic_panel(n_stocks=8, n_days=80)
        )
        selected = frame[frame["label"].notna()]
        self.assertTrue((selected["date"] == prediction_date).all())
        self.assertEqual(len(selected), 8)
        self.assertEqual(coverage["coverage_ratio"], 1.0)

    def test_saved_model_path_only_calls_predict(self):
        artifact = Path("unused-test-artifact")
        with patch(
            "asharerank.prediction.inference.engine.latest_saved_artifact",
            return_value=artifact,
        ), patch(
            "asharerank.prediction.inference.engine.joblib.load",
            side_effect=[_InferenceOnlyAdapter(), _InferenceOnlyModel()],
        ):
            outputs, errors, coverage = predict_saved_models(
                synthetic_panel(n_stocks=8, n_days=80), ["ridge"]
            )
        self.assertFalse(errors)
        self.assertEqual([output.model_name for output in outputs], ["ridge"])
        self.assertEqual(len(outputs[0].predictions), 8)
        self.assertEqual(coverage["models_completed"], 1)

    def test_live_request_refreshes_data_by_default(self):
        request = LivePredictRequest()
        self.assertTrue(request.refresh_data)
        self.assertIn("lite_rank", request.models)


if __name__ == "__main__":
    unittest.main()
