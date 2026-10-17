from __future__ import annotations

from typing import Protocol

import numpy as np


class ThresholdStrategy(Protocol):
    def choose(
        self,
        *,
        train_scores: np.ndarray,
        validation_scores: np.ndarray,
        validation_labels: np.ndarray,
        test_scores: np.ndarray,
    ) -> float: ...
