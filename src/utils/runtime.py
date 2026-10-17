from __future__ import annotations

import logging
import os
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

BYTES_PER_GIGABYTE = 1024**3
CPU_DEVICE = "cpu"
VISIBLE_DEVICES_VARIABLE = "CUDA_VISIBLE_DEVICES"


class DeviceMemoryError(RuntimeError):
    pass


def set_seed(*, seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(*, preferred: str) -> "torch.device":
    import torch

    if preferred.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        return torch.device(CPU_DEVICE)
    device = torch.device(preferred)
    if device.type == "cuda" and device.index is None:
        return torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    return device


def ensure_free_memory(
    *,
    device: "torch.device",
    minimum_gigabytes: float,
) -> None:
    import torch

    if device.type != "cuda" or minimum_gigabytes <= 0:
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gigabytes = free_bytes / BYTES_PER_GIGABYTE
    logger.info(
        "GPU %s (server id %s, %s) has %.1f/%.1f GB free",
        device,
        os.environ.get(VISIBLE_DEVICES_VARIABLE, "all"),
        torch.cuda.get_device_name(device),
        free_gigabytes,
        total_bytes / BYTES_PER_GIGABYTE,
    )
    if free_gigabytes < minimum_gigabytes:
        server_id = os.environ.get(VISIBLE_DEVICES_VARIABLE, "all")
        raise DeviceMemoryError(
            f"GPU server id {server_id} has only {free_gigabytes:.1f} GB free, need {minimum_gigabytes:.1f} GB. "
            "Pick another GPU with GPU_IDS in the script"
        )


def count_parameters(*, module: "torch.nn.Module") -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def module_size_megabytes(*, module: "torch.nn.Module") -> float:
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())
    return (parameter_bytes + buffer_bytes) / 1024**2
