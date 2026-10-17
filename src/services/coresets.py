from __future__ import annotations

import numpy as np
import torch

from src.data.classes.enums import CoresetKind
from src.interface.coreset import Coreset


class StreamCore:
    def __init__(
        self,
        *,
        capacity: int,
        generator: np.random.Generator,
        device: torch.device,
    ) -> None:
        self._capacity = capacity
        self._generator = generator
        self._device = device
        self._features: torch.Tensor | None = None
        self._indices: list[int] = []
        self.replacements = 0

    def _append(
        self,
        *,
        feature: torch.Tensor,
        index: int,
    ) -> None:
        if self._features is None:
            self._features = torch.empty(self._capacity, feature.numel(), device=self._device)
        self._features[len(self._indices)] = feature
        self._indices.append(index)

    def _nearest_distance(
        self,
        *,
        feature: torch.Tensor,
        excluded: int,
    ) -> float:
        distances = torch.cdist(feature.unsqueeze(0), self._features).squeeze(0)
        distances[excluded] = torch.inf
        return float(distances.min())

    def _offer(
        self,
        *,
        feature: torch.Tensor,
        index: int,
    ) -> None:
        if len(self._indices) < self._capacity:
            self._append(
                feature=feature,
                index=index,
            )
            return
        candidate = int(self._generator.integers(self._capacity))
        kept_spread = self._nearest_distance(
            feature=self._features[candidate],
            excluded=candidate,
        )
        new_spread = self._nearest_distance(
            feature=feature,
            excluded=candidate,
        )
        if new_spread > kept_spread:
            self._features[candidate] = feature
            self._indices[candidate] = index
            self.replacements += 1

    def insert(
        self,
        *,
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        flattened = features.reshape(features.shape[0], -1).to(self._device, dtype=torch.float32)
        for feature, index in zip(flattened, indices.tolist()):
            self._offer(
                feature=feature,
                index=int(index),
            )

    def selected(self) -> list[int]:
        return list(self._indices)


class ReservoirBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        generator: np.random.Generator,
    ) -> None:
        self._capacity = capacity
        self._generator = generator
        self._indices: list[int] = []
        self._seen = 0

    def insert(
        self,
        *,
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        for index in indices.tolist():
            self._seen += 1
            if len(self._indices) < self._capacity:
                self._indices.append(int(index))
                continue
            slot = int(self._generator.integers(self._seen))
            if slot < self._capacity:
                self._indices[slot] = int(index)

    def selected(self) -> list[int]:
        return list(self._indices)


def build_coreset(
    *,
    kind: str,
    capacity: int,
    seed: int,
    device: torch.device,
) -> Coreset:
    generator = np.random.default_rng(seed)
    builders = {
        CoresetKind.STREAMCORE: lambda: StreamCore(
            capacity=capacity,
            generator=generator,
            device=device,
        ),
        CoresetKind.RANDOM: lambda: ReservoirBuffer(
            capacity=capacity,
            generator=generator,
        ),
    }
    return builders[CoresetKind(kind)]()
