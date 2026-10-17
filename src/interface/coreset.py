from __future__ import annotations

from typing import Protocol

import torch


class Coreset(Protocol):
    def insert(
        self,
        *,
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> None: ...

    def selected(self) -> list[int]: ...
