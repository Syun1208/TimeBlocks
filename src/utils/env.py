from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_FILE_NAME = ".env"
STORAGE_ROOT_VARIABLE = "TIMEBLOCKS_STORAGE_ROOT"
HF_HOME_VARIABLE = "HF_HOME"
DATA_DIRECTORY = "data"
LOGS_DIRECTORY = "logs"
WEIGHTS_DIRECTORY = "weights"
OUTPUTS_DIRECTORY = "outputs"
HF_CACHE_DIRECTORY = "hf_cache"
THIRD_PARTY_DIRECTORY = "third_party"
MISSING_STORAGE_MESSAGE = (
    f"{STORAGE_ROOT_VARIABLE} is not set. " f"Add it to the {ENV_FILE_NAME} file of the parent folder, see .env.example"
)


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StoragePaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / DATA_DIRECTORY

    @property
    def logs(self) -> Path:
        return self.root / LOGS_DIRECTORY

    @property
    def weights(self) -> Path:
        return self.root / WEIGHTS_DIRECTORY

    @property
    def outputs(self) -> Path:
        return self.root / OUTPUTS_DIRECTORY

    @property
    def hf_cache(self) -> Path:
        return self.root / HF_CACHE_DIRECTORY

    @property
    def third_party(self) -> Path:
        return self.root / THIRD_PARTY_DIRECTORY


def find_env_file(*, start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / ENV_FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def load_env(*, start: Path) -> Path | None:
    from dotenv import load_dotenv

    env_file = find_env_file(start=start)
    if env_file is not None:
        load_dotenv(
            dotenv_path=env_file,
            override=False,
        )
    return env_file


def resolve_storage(*, start: Path) -> StoragePaths:
    load_env(start=start)
    root = os.environ.get(STORAGE_ROOT_VARIABLE)
    if not root:
        raise StorageError(MISSING_STORAGE_MESSAGE)
    storage = StoragePaths(root=Path(root).expanduser().resolve())
    os.environ.setdefault(
        HF_HOME_VARIABLE,
        str(storage.hf_cache),
    )
    return storage
