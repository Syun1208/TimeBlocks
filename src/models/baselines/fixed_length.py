from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_length(
    *,
    x: torch.Tensor,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, current, channels = x.shape
    if current > length:
        resized = F.interpolate(
            x.transpose(1, 2),
            size=length,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        return resized, torch.ones(batch, length, device=x.device)
    padding = length - current
    padded = torch.cat([torch.zeros(batch, padding, channels, device=x.device), x], dim=1)
    observed = torch.ones(batch, length, device=x.device)
    observed[:, :padding] = 0.0
    return padded, observed


def fit_mask(
    *,
    mask: torch.Tensor,
    length: int,
) -> torch.Tensor:
    batch, current, _ = mask.shape
    if current >= length:
        return mask[:, -length:, 0]
    return torch.cat([torch.ones(batch, length - current, device=mask.device), mask[..., 0]], dim=1)
