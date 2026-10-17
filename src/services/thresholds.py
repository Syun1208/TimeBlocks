from __future__ import annotations

import numpy as np

from src.data.classes.enums import ThresholdKind
from src.data.classes.settings import AnomalySettings
from src.interface.threshold import ThresholdStrategy
from src.utils.metrics import search_best_threshold

PERCENT = 100.0


class ValidationF1Threshold:
    def __init__(self, *, point_adjustment: bool) -> None:
        self._point_adjustment = point_adjustment

    def choose(
        self,
        *,
        train_scores: np.ndarray,
        validation_scores: np.ndarray,
        validation_labels: np.ndarray,
        test_scores: np.ndarray,
    ) -> float:
        return search_best_threshold(
            scores=validation_scores,
            labels=validation_labels,
            point_adjustment=self._point_adjustment,
        ).threshold


class PercentileThreshold:
    def __init__(self, *, anomaly_ratio_percent: float) -> None:
        self._anomaly_ratio_percent = anomaly_ratio_percent

    def choose(
        self,
        *,
        train_scores: np.ndarray,
        validation_scores: np.ndarray,
        validation_labels: np.ndarray,
        test_scores: np.ndarray,
    ) -> float:
        reference = np.concatenate([train_scores, test_scores])
        return float(np.percentile(reference, PERCENT - self._anomaly_ratio_percent))


def build_threshold_strategy(*, settings: AnomalySettings) -> ThresholdStrategy:
    builders = {
        ThresholdKind.VALIDATION_F1: lambda: ValidationF1Threshold(point_adjustment=settings.point_adjustment),
        ThresholdKind.PERCENTILE: lambda: PercentileThreshold(anomaly_ratio_percent=settings.anomaly_ratio_percent),
    }
    return builders[ThresholdKind(settings.threshold)]()
