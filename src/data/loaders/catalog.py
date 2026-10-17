from __future__ import annotations

from dataclasses import dataclass

from src.data.classes.enums import Task

THUML_REPOSITORY = "thuml/Time-Series-Library"
UCR_ARCHIVE_URL = "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
FORECASTING_DIRECTORY = "forecasting"
ANOMALY_DIRECTORY = "anomaly_detection"
CLASSIFICATION_DIRECTORY = "classification"
ALL_UCR = "all"


class UnknownDatasetError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class RemoteFile:
    remote_path: str
    local_path: str


@dataclass(frozen=True, slots=True)
class SplitCounts:
    train: int
    validation: int
    test: int


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    task: Task
    reader: str
    files: tuple[RemoteFile, ...]
    split_counts: SplitCounts | None = None


def _forecasting(
    *,
    name: str,
    remote_path: str,
    counts: SplitCounts,
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        task=Task.FORECASTING,
        reader="csv_forecasting",
        files=(
            RemoteFile(
                remote_path=remote_path,
                local_path=f"{FORECASTING_DIRECTORY}/{name}.csv",
            ),
        ),
        split_counts=counts,
    )


def _anomaly_npy(*, name: str) -> DatasetSpec:
    suffixes = ("train", "test", "test_label")
    return DatasetSpec(
        name=name,
        task=Task.ANOMALY_DETECTION,
        reader="npy_anomaly",
        files=tuple(
            RemoteFile(
                remote_path=f"{name}/{name}_{suffix}.npy",
                local_path=f"{ANOMALY_DIRECTORY}/{name}/{name}_{suffix}.npy",
            )
            for suffix in suffixes
        ),
    )


HOURLY_COUNTS = SplitCounts(train=10452, validation=3484, test=3484)
MINUTE_COUNTS = SplitCounts(train=41808, validation=13936, test=13936)
WEATHER_COUNTS = SplitCounts(train=24388, validation=14154, test=14154)

DATASETS: dict[str, DatasetSpec] = {
    "ETTh1": _forecasting(name="ETTh1", remote_path="ETT-small/ETTh1.csv", counts=HOURLY_COUNTS),
    "ETTh2": _forecasting(name="ETTh2", remote_path="ETT-small/ETTh2.csv", counts=HOURLY_COUNTS),
    "ETTm1": _forecasting(name="ETTm1", remote_path="ETT-small/ETTm1.csv", counts=MINUTE_COUNTS),
    "ETTm2": _forecasting(name="ETTm2", remote_path="ETT-small/ETTm2.csv", counts=MINUTE_COUNTS),
    "Weather": _forecasting(name="Weather", remote_path="weather/weather.csv", counts=WEATHER_COUNTS),
    "SMD": _anomaly_npy(name="SMD"),
    "MSL": _anomaly_npy(name="MSL"),
    "SMAP": _anomaly_npy(name="SMAP"),
    "SWaT": DatasetSpec(
        name="SWaT",
        task=Task.ANOMALY_DETECTION,
        reader="csv_swat",
        files=(
            RemoteFile(
                remote_path="SWaT/swat_train2.csv",
                local_path=f"{ANOMALY_DIRECTORY}/SWaT/swat_train2.csv",
            ),
            RemoteFile(
                remote_path="SWaT/swat2.csv",
                local_path=f"{ANOMALY_DIRECTORY}/SWaT/swat2.csv",
            ),
        ),
    ),
}


def ucr_spec(*, name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        task=Task.CLASSIFICATION,
        reader="ts_ucr",
        files=(
            RemoteFile(
                remote_path=UCR_ARCHIVE_URL.format(name=name),
                local_path=f"{CLASSIFICATION_DIRECTORY}/UCR/{name}",
            ),
        ),
    )


def ucr_names() -> tuple[str, ...]:
    from aeon.datasets.tsc_datasets import univariate

    return tuple(sorted(univariate))


def resolve_spec(*, name: str) -> DatasetSpec:
    if name in DATASETS:
        return DATASETS[name]
    if name in ucr_names():
        return ucr_spec(name=name)
    raise UnknownDatasetError(f"Unknown dataset {name!r}")


def expand_ucr(*, names: tuple[str, ...]) -> tuple[str, ...]:
    return ucr_names() if ALL_UCR in names else names
