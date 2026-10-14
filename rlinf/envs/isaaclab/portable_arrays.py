# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Portable CPU arrays for trusted, node-local simulator subprocesses.

Do not pickle CUDA storage or NumPy implementation classes across Python,
Torch and NumPy versions. Only explicit dtype/shape/bytes cross this boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def encode(value: Any) -> Any:
    """Copy tensor leaves to a version-neutral CPU wire representation."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Simulator IPC does not admit object arrays")
        array = np.ascontiguousarray(value)
        return {"__ndarray_v1__": (array.dtype.str, array.shape, array.tobytes())}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(encode(item) for item in value)
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(f"Unsupported simulator IPC leaf: {type(value).__name__}")


def decode(value: Any, *, device: str | None = None) -> Any:
    """Decode independent arrays, optionally materializing Torch tensors."""
    if isinstance(value, dict) and set(value) == {"__ndarray_v1__"}:
        dtype, shape, payload = value["__ndarray_v1__"]
        dtype = np.dtype(dtype)
        if dtype.hasobject:
            raise TypeError("Simulator IPC does not admit object arrays")
        array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        if device is not None:
            import torch

            return torch.from_numpy(array).to(device=device)
        return array
    if isinstance(value, dict):
        return {key: decode(item, device=device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(decode(item, device=device) for item in value)
    if isinstance(value, list):
        return [decode(item, device=device) for item in value]
    return value
