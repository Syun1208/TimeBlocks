from __future__ import annotations

import numpy as np

from src.data.classes.series import SeriesSplits

GLUONTS_FREQUENCY = "S"


class TimeFeatureEncoder:
    def __init__(self, *, frequency: str = GLUONTS_FREQUENCY) -> None:
        self._frequency = frequency
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, *, series: SeriesSplits) -> np.ndarray:
        import pandas as pd
        from gluonts.time_feature import time_features_from_frequency_str

        if series.timestamps is None:
            raise ValueError(f"{series.name} has no timestamps for time features")
        if series.name not in self._cache:
            index = pd.DatetimeIndex(series.timestamps)
            features = [feature(index) for feature in time_features_from_frequency_str(self._frequency)]
            self._cache[series.name] = np.stack(features, axis=1).astype(np.float32)
        return self._cache[series.name]
