from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

CALIBRATED_DIRECTORY = "calibrated"
BASELINE_DIRECTORY = "baselines"
MODEL_FILE = "model.pt"
METADATA_FILE = "metadata.json"
IDENTITY_FILE = "identity.json"
DIGEST_LENGTH = 12
TEMPORARY_SUFFIX = ".partial"


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CheckpointKey:
    group: str
    name: str
    identity: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.identity,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:DIGEST_LENGTH]


def trainable_state(*, model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    buffers = {name for name, _ in model.named_buffers()}
    return {
        name: value.detach().cpu() for name, value in model.state_dict().items() if name in trainable or name in buffers
    }


def write_atomically(
    *,
    path: Path,
    writer: Any,
) -> None:
    temporary = path.with_name(path.name + TEMPORARY_SUFFIX)
    writer(temporary)
    os.replace(temporary, path)


class CheckpointStore:
    def __init__(self, *, root: Path) -> None:
        self._root = root

    def directory(self, *, key: CheckpointKey) -> Path:
        return self._root / key.group / key.digest / key.name

    def exists(self, *, key: CheckpointKey) -> bool:
        directory = self.directory(key=key)
        return (directory / MODEL_FILE).exists() and (directory / METADATA_FILE).exists()

    def save(
        self,
        *,
        key: CheckpointKey,
        model: nn.Module,
        metadata: dict[str, Any],
    ) -> Path:
        directory = self.directory(key=key)
        directory.mkdir(parents=True, exist_ok=True)
        identity_file = directory.parent / IDENTITY_FILE
        if not identity_file.exists():
            write_atomically(
                path=identity_file,
                writer=lambda path: path.write_text(
                    json.dumps(key.identity, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                ),
            )
        write_atomically(
            path=directory / MODEL_FILE,
            writer=lambda path: torch.save(trainable_state(model=model), path),
        )
        write_atomically(
            path=directory / METADATA_FILE,
            writer=lambda path: path.write_text(
                json.dumps(metadata, indent=2, default=str),
                encoding="utf-8",
            ),
        )
        logger.info("Saved checkpoint %s", directory)
        return directory

    def load(
        self,
        *,
        key: CheckpointKey,
        model: nn.Module,
    ) -> dict[str, Any]:
        directory = self.directory(key=key)
        state = torch.load(
            directory / MODEL_FILE,
            map_location="cpu",
            weights_only=True,
        )
        result = model.load_state_dict(
            state,
            strict=False,
        )
        if result.unexpected_keys:
            raise CheckpointError(f"Checkpoint {directory} does not match the model: {result.unexpected_keys[:5]}")
        logger.info("Loaded checkpoint %s", directory)
        return json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
