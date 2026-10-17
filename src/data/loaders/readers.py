from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data.classes.series import ClassificationSplits, SeriesSplits
from src.data.loaders.catalog import DatasetSpec, resolve_spec

logger = logging.getLogger(__name__)

DATE_COLUMN = "date"
MINIMUM_STANDARD_DEVIATION = 1.0e-8
VALUE_DTYPE = np.float32


class DatasetMissingError(FileNotFoundError):
    pass


def standardize_with_train(
    *,
    values: np.ndarray,
    train_end: int,
) -> np.ndarray:
    train = values[:train_end]
    mean = train.mean(axis=0, keepdims=True)
    deviation = train.std(axis=0, keepdims=True)
    deviation = np.where(deviation < MINIMUM_STANDARD_DEVIATION, 1.0, deviation)
    return ((values - mean) / deviation).astype(VALUE_DTYPE)


def require_files(*, paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise DatasetMissingError(f"Missing dataset files {missing}. Run: bash scripts/download_data.sh")


class CsvForecastingReader:
    def read(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> SeriesSplits:
        import pandas as pd

        path = data_root / spec.files[0].local_path
        require_files(paths=[path])
        frame = pd.read_csv(path)
        timestamps = pd.to_datetime(frame.pop(DATE_COLUMN)).to_numpy()
        values = frame.to_numpy(dtype=np.float64)
        counts = spec.split_counts
        expected = counts.train + counts.validation + counts.test
        if expected != len(values):
            logger.warning(
                "%s has %d rows but Table A1 lists %d, trimming to Table A1", spec.name, len(values), expected
            )
            values = values[:expected]
            timestamps = timestamps[:expected]
        return SeriesSplits(
            name=spec.name,
            values=standardize_with_train(
                values=values,
                train_end=counts.train,
            ),
            train_end=counts.train,
            validation_end=counts.train + counts.validation,
            timestamps=timestamps,
        )


def anomaly_splits(
    *,
    name: str,
    train: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
) -> SeriesSplits:
    values = np.nan_to_num(np.concatenate([train, test], axis=0).astype(np.float64))
    labels = np.concatenate([np.zeros(len(train), dtype=np.int8), test_labels.reshape(-1).astype(np.int8)])
    train_end = len(train)
    return SeriesSplits(
        name=name,
        values=standardize_with_train(
            values=values,
            train_end=train_end,
        ),
        train_end=train_end,
        validation_end=train_end + len(test) // 2,
        labels=labels,
    )


class NpyAnomalyReader:
    def read(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> SeriesSplits:
        train_path, test_path, label_path = (data_root / remote.local_path for remote in spec.files)
        require_files(paths=[train_path, test_path, label_path])
        return anomaly_splits(
            name=spec.name,
            train=np.load(train_path),
            test=np.load(test_path),
            test_labels=np.load(label_path),
        )


class SwatCsvReader:
    def read(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> SeriesSplits:
        import pandas as pd

        train_path, test_path = (data_root / remote.local_path for remote in spec.files)
        require_files(paths=[train_path, test_path])
        train = pd.read_csv(train_path).to_numpy(dtype=np.float64)
        test = pd.read_csv(test_path).to_numpy(dtype=np.float64)
        return anomaly_splits(
            name=spec.name,
            train=train[:, :-1],
            test=test[:, :-1],
            test_labels=test[:, -1],
        )


def fill_and_resample(
    *,
    series: np.ndarray,
    length: int,
) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(length, dtype=VALUE_DTYPE)
    positions = np.arange(len(values))
    values = np.interp(positions, positions[valid], values[valid])
    if len(values) == length:
        return values.astype(VALUE_DTYPE)
    return np.interp(np.linspace(0, len(values) - 1, length), positions, values).astype(VALUE_DTYPE)


class UcrReader:
    def _load(self, *, path: Path) -> tuple[list[np.ndarray], np.ndarray]:
        from aeon.datasets import load_from_ts_file

        inputs, labels = load_from_ts_file(str(path))
        return [np.asarray(item)[0] for item in inputs], np.asarray(labels)

    def read(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> ClassificationSplits:
        directory = data_root / spec.files[0].local_path
        train_path = directory / f"{spec.name}_TRAIN.ts"
        test_path = directory / f"{spec.name}_TEST.ts"
        require_files(paths=[train_path, test_path])
        train_series, train_labels = self._load(path=train_path)
        test_series, test_labels = self._load(path=test_path)
        length = max(len(series) for series in (*train_series, *test_series))
        class_names = tuple(sorted({*map(str, train_labels), *map(str, test_labels)}))
        lookup = {name: index for index, name in enumerate(class_names)}

        def stack(*, series_list: list[np.ndarray]) -> np.ndarray:
            return np.stack(
                [
                    fill_and_resample(
                        series=series,
                        length=length,
                    )
                    for series in series_list
                ]
            )

        return ClassificationSplits(
            name=spec.name,
            train_inputs=stack(series_list=train_series),
            train_labels=np.array([lookup[str(label)] for label in train_labels], dtype=np.int64),
            test_inputs=stack(series_list=test_series),
            test_labels=np.array([lookup[str(label)] for label in test_labels], dtype=np.int64),
            class_names=class_names,
        )


READERS = {
    "csv_forecasting": CsvForecastingReader(),
    "npy_anomaly": NpyAnomalyReader(),
    "csv_swat": SwatCsvReader(),
    "ts_ucr": UcrReader(),
}


class DatasetRepository:
    def __init__(self, *, data_root: Path) -> None:
        self._data_root = data_root
        self._cache: dict[str, SeriesSplits | ClassificationSplits] = {}

    def load(self, *, name: str) -> SeriesSplits | ClassificationSplits:
        if name not in self._cache:
            spec = resolve_spec(name=name)
            logger.debug("Loading dataset %s with reader %s", name, spec.reader)
            self._cache[name] = READERS[spec.reader].read(
                spec=spec,
                data_root=self._data_root,
            )
        return self._cache[name]

    def load_series(self, *, name: str) -> SeriesSplits:
        loaded = self.load(name=name)
        if not isinstance(loaded, SeriesSplits):
            raise TypeError(f"{name} is not a continuous time series dataset")
        return loaded

    def load_classification(self, *, name: str) -> ClassificationSplits:
        loaded = self.load(name=name)
        if not isinstance(loaded, ClassificationSplits):
            raise TypeError(f"{name} is not a classification dataset")
        return loaded
