"""统一单模型训练与多模型独立实验入口。"""

from asharerank.training.trainer import ExperimentResult, run_experiment, run_experiments

__all__ = ["ExperimentResult", "run_experiment", "run_experiments"]
