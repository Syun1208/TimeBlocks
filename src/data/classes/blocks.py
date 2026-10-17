from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelRecord:
    model_id: str
    task: str
    dataset: str
    architecture: str
    context_length: int
    horizon: int
    num_blocks: int
    output_size: int
    validation_loss: float
    num_parameters: int

    def block_id(self, *, position: int) -> str:
        return f"{self.model_id}/b{position}"

    def sampling_stride(self, *, position: int) -> int:
        return 2 ** (self.num_blocks - position)


@dataclass(frozen=True, slots=True)
class BlockRecord:
    block_id: str
    model_id: str
    position: int
    task: str
    dataset: str
    architecture: str
    context_length: int
    sampling_stride: int


@dataclass(frozen=True, slots=True)
class ClusterIndex:
    labels: tuple[int, ...]
    representatives: tuple[int, ...]
    context_lengths: tuple[int, ...]

    @property
    def num_clusters(self) -> int:
        return len(self.representatives)

    def members(self, *, cluster: int) -> list[int]:
        return [index for index, label in enumerate(self.labels) if label == cluster]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    cluster: int
    context_length: int
    block_ids: tuple[str, ...]
    cluster_distances: tuple[float, ...]
    block_distances: tuple[float, ...]
    seconds: float
