from __future__ import annotations

import logging
import shutil
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

from src.data.loaders.catalog import THUML_REPOSITORY, DatasetSpec, RemoteFile

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120
PARTIAL_SUFFIX = ".part"


class DownloadError(RuntimeError):
    pass


class HuggingFaceDatasetDownloader:
    def __init__(self, *, repository: str = THUML_REPOSITORY) -> None:
        self._repository = repository

    def download(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> list[Path]:
        from huggingface_hub import hf_hub_download

        paths = []
        for remote in spec.files:
            target = data_root / remote.local_path
            if target.exists():
                logger.debug("Already present: %s", target)
                paths.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(
                repo_id=self._repository,
                filename=remote.remote_path,
                repo_type="dataset",
            )
            shutil.copyfile(
                cached,
                target,
            )
            logger.info("Saved %s", target)
            paths.append(target)
        return paths


class UcrArchiveDownloader:
    def __init__(
        self,
        *,
        retries: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._retries = retries
        self._sleep = sleep

    def _fetch(
        self,
        *,
        remote: RemoteFile,
        archive: Path,
    ) -> None:
        import requests

        partial = archive.with_suffix(archive.suffix + PARTIAL_SUFFIX)
        with requests.get(
            remote.remote_path,
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    handle.write(chunk)
        partial.rename(archive)

    def download(
        self,
        *,
        spec: DatasetSpec,
        data_root: Path,
    ) -> list[Path]:
        remote = spec.files[0]
        directory = data_root / remote.local_path
        train_file = directory / f"{spec.name}_TRAIN.ts"
        if train_file.exists():
            return [directory]
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / f"{spec.name}.zip"
        for attempt in range(1, self._retries + 1):
            try:
                self._fetch(
                    remote=remote,
                    archive=archive,
                )
                break
            except Exception as error:
                logger.warning("Download %s failed (attempt %d): %s", spec.name, attempt, error)
                if attempt == self._retries:
                    raise DownloadError(f"Could not download {spec.name}") from error
                self._sleep(2.0 * attempt)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if member.endswith(".ts"):
                    (directory / Path(member).name).write_bytes(bundle.read(member))
        archive.unlink()
        return [directory]
