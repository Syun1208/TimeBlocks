from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.classes.blocks import BlockRecord, ClusterIndex, ModelRecord
from src.data.classes.enums import Architecture
from src.data.classes.settings import BlockSettings, RouterSettings
from src.models.timeblocks.block import TimeBlock
from src.models.timeblocks.block_model import BlockModel
from src.models.timeblocks.heads import ResampleHead
from src.models.timeblocks.router import Router
from src.utils.config_loader import build_dataclass

logger = logging.getLogger(__name__)

BLOCKBASE_DIRECTORY = "blockbase"
INDEX_DIRECTORY = "blockbase_index"
MODEL_FILE = "model.pt"
RECORD_FILE = "record.json"
ROUTER_FILE = "router.pt"
FINGERPRINTS_FILE = "fingerprints.npy"
BLOCKS_FILE = "blocks.json"
CLUSTERS_FILE = "clusters.json"
BLOCK_PREFIX = "blocks"
NO_EXCLUSION = "none"


class BlockbaseError(RuntimeError):
    pass


def block_records_of(*, record: ModelRecord) -> list[BlockRecord]:
    return [
        BlockRecord(
            block_id=record.block_id(position=position),
            model_id=record.model_id,
            position=position,
            task=record.task,
            dataset=record.dataset,
            architecture=record.architecture,
            context_length=record.context_length,
            sampling_stride=record.sampling_stride(position=position),
        )
        for position in range(1, record.num_blocks + 1)
    ]


class BlockbaseStore:
    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._state_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._settings_cache: dict[str, BlockSettings] = {}

    @property
    def root(self) -> Path:
        return self._root

    def model_directory(self, *, model_id: str) -> Path:
        return self._root / model_id

    def exists(self, *, model_id: str) -> bool:
        return (self.model_directory(model_id=model_id) / RECORD_FILE).exists()

    def save(
        self,
        *,
        record: ModelRecord,
        model: BlockModel,
        settings: BlockSettings,
    ) -> Path:
        directory = self.model_directory(model_id=record.model_id)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            directory / MODEL_FILE,
        )
        payload = {"record": asdict(record), "block_settings": asdict(settings)}
        (directory / RECORD_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return directory

    def _payload(self, *, model_id: str) -> dict:
        return json.loads((self.model_directory(model_id=model_id) / RECORD_FILE).read_text(encoding="utf-8"))

    def records(
        self,
        *,
        tasks: Sequence[str],
        excluded_datasets: Sequence[str],
    ) -> list[ModelRecord]:
        if not self._root.exists():
            raise BlockbaseError(f"Blockbase {self._root} does not exist, run the pretrain step first")
        records = []
        for record_file in sorted(self._root.glob(f"*/{RECORD_FILE}")):
            record = ModelRecord(**json.loads(record_file.read_text(encoding="utf-8"))["record"])
            if record.task in tasks and record.dataset not in excluded_datasets:
                records.append(record)
        return records

    def block_settings(self, *, model_id: str) -> BlockSettings:
        if model_id not in self._settings_cache:
            self._settings_cache[model_id] = build_dataclass(
                settings_type=BlockSettings,
                values=self._payload(model_id=model_id)["block_settings"],
            )
        return self._settings_cache[model_id]

    def _state(self, *, model_id: str) -> dict[str, torch.Tensor]:
        if model_id not in self._state_cache:
            self._state_cache[model_id] = torch.load(
                self.model_directory(model_id=model_id) / MODEL_FILE,
                map_location="cpu",
                weights_only=True,
            )
        return self._state_cache[model_id]

    def load_block(self, *, block: BlockRecord) -> TimeBlock:
        module = TimeBlock(
            settings=self.block_settings(model_id=block.model_id),
            architecture=Architecture(block.architecture),
            sampling_stride=block.sampling_stride,
        )
        prefix = f"{BLOCK_PREFIX}.{block.position - 1}."
        state = self._state(model_id=block.model_id)
        module.load_state_dict({key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)})
        return module

    def load_backbone(self, *, record: ModelRecord) -> BlockModel:
        settings = self.block_settings(model_id=record.model_id)
        return BlockModel(
            settings=settings,
            blocks=[self.load_block(block=block) for block in block_records_of(record=record)],
            head=ResampleHead(output_length=settings.common_length),
        )

    def release(self) -> None:
        self._state_cache.clear()


class BlockbaseIndexStore:
    def __init__(self, *, root: Path) -> None:
        self._root = root

    def directory(
        self,
        *,
        blockbase_name: str,
        pool_tasks: Sequence[str],
        excluded_dataset: str | None,
    ) -> Path:
        tasks = "+".join(sorted(pool_tasks))
        return self._root / blockbase_name / tasks / f"without_{excluded_dataset or NO_EXCLUSION}"

    def save(
        self,
        *,
        directory: Path,
        router: Router,
        router_settings: RouterSettings,
        fingerprints: np.ndarray,
        blocks: Sequence[BlockRecord],
        clusters: ClusterIndex,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": router.state_dict(),
                "settings": asdict(router_settings),
                "fingerprint_size": int(fingerprints.shape[1]),
            },
            directory / ROUTER_FILE,
        )
        np.save(directory / FINGERPRINTS_FILE, fingerprints)
        (directory / BLOCKS_FILE).write_text(
            json.dumps([asdict(block) for block in blocks], indent=2), encoding="utf-8"
        )
        (directory / CLUSTERS_FILE).write_text(json.dumps(asdict(clusters), indent=2), encoding="utf-8")

    def load(self, *, directory: Path) -> tuple[Router, np.ndarray, list[BlockRecord], ClusterIndex]:
        if not (directory / ROUTER_FILE).exists():
            raise BlockbaseError(f"Blockbase index {directory} not found, run the blockbase step first")
        payload = torch.load(directory / ROUTER_FILE, map_location="cpu", weights_only=True)
        router = Router(
            settings=build_dataclass(
                settings_type=RouterSettings,
                values=payload["settings"],
            ),
            fingerprint_size=payload["fingerprint_size"],
        )
        router.load_state_dict(payload["state"])
        blocks = [BlockRecord(**item) for item in json.loads((directory / BLOCKS_FILE).read_text(encoding="utf-8"))]
        cluster_values = json.loads((directory / CLUSTERS_FILE).read_text(encoding="utf-8"))
        clusters = ClusterIndex(**{key: tuple(value) for key, value in cluster_values.items()})
        return router, np.load(directory / FINGERPRINTS_FILE), blocks, clusters


def freeze(*, module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module
