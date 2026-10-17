from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.data.classes.series import ClassificationSplits, SeriesSplits
from src.data.loaders.catalog import DatasetSpec


class Downloader(Protocol):
    def download(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> list[Path]: ...


class DatasetReader(Protocol):
    def read(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> SeriesSplits | ClassificationSplits: ...
