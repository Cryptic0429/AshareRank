"""模型训练和推理耗时记录。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class Timer:
    seconds: float = 0.0


@contextmanager
def measure_time():
    timer = Timer()
    start = perf_counter()
    try:
        yield timer
    finally:
        timer.seconds = perf_counter() - start
