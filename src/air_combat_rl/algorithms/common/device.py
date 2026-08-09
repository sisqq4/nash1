"""PyTorch device selection and reproducible seeding."""

from __future__ import annotations

import random

import numpy as np


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "the torch backend requires: python -m pip install -e '.[torch]'"
        ) from exc
    return torch


def resolve_device(requested: str = "auto"):
    torch = require_torch()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} requested but CUDA is unavailable"
        )
    return device


def seed_everything(seed: int, deterministic: bool = False) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic


def describe_device(device) -> dict:
    torch = require_torch()
    result = {
        "resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["capability"] = list(torch.cuda.get_device_capability(device))
    return result
