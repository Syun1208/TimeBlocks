from __future__ import annotations

import logging
from pathlib import Path

from src.data.classes.enums import Task
from src.data.classes.settings import DownloadSettings
from src.data.loaders.catalog import expand_ucr, resolve_spec
from src.interface.dataset_io import Downloader
from src.utils.progress import DOWNLOAD, finish, progress

logger = logging.getLogger(__name__)


class DatasetDownloadService:
    def __init__(
        self,
        *,
        settings: DownloadSettings,
        downloaders: dict[Task, Downloader],
        data_root: Path,
    ) -> None:
        self._settings = settings
        self._downloaders = downloaders
        self._data_root = data_root

    def names(self) -> list[str]:
        return [
            *self._settings.forecasting,
            *self._settings.anomaly_detection,
            *expand_ucr(names=self._settings.classification),
        ]

    def run(self) -> list[Path]:
        names = self.names()
        bar = progress(
            stage=DOWNLOAD,
            total=len(names),
            unit="dataset",
        )
        saved: list[Path] = []
        failed: list[str] = []
        for name in names:
            bar.set_postfix_str(name)
            spec = resolve_spec(name=name)
            try:
                saved.extend(
                    self._downloaders[spec.task].download(
                        spec=spec,
                        data_root=self._data_root,
                    )
                )
            except Exception:
                logger.exception("Failed to download %s", name)
                failed.append(name)
            bar.update(1)
        finish(bar=bar)
        if failed:
            logger.warning("Failed datasets: %s", failed)
        logger.info("Datasets ready under %s (%d paths)", self._data_root, len(saved))
        return saved
