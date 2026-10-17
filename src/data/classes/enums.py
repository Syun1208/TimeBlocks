from __future__ import annotations

from enum import Enum


class Task(str, Enum):
    FORECASTING = "forecasting"
    IMPUTATION = "imputation"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"


class Architecture(str, Enum):
    MLP = "mlp"
    LSTM = "lstm"
    ATTENTION = "attention"


class CoresetKind(str, Enum):
    STREAMCORE = "streamcore"
    RANDOM = "random"


class ThresholdKind(str, Enum):
    VALIDATION_F1 = "validation_f1"
    PERCENTILE = "percentile"


class RouterLoss(str, Enum):
    MSE = "mse"
    COSINE = "cosine"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
