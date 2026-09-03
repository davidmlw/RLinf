# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent-buffer TensorRT wrapper compatible with GR00T's Python runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any


class PersistentEngine:
    """Execute a TensorRT plan on the current PyTorch stream without host sync."""

    def __init__(self, file: str, plugins: list[str] | None = None):
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        self.file = str(Path(file).resolve(strict=True))
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.plugins = [
            ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in (plugins or [])
        ]
        runtime = trt.Runtime(self.logger)
        self.handle = runtime.deserialize_cuda_engine(Path(self.file).read_bytes())
        if self.handle is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.file}")
        self.execution_context = self.handle.create_execution_context()
        if self.execution_context is None:
            raise RuntimeError(f"failed to create TensorRT context: {self.file}")
        self.in_meta = []
        self.out_meta = []
        for name in self.handle:
            item = [
                name,
                tuple(self.handle.get_tensor_shape(name)),
                self._torch_type(self.handle.get_tensor_dtype(name)),
            ]
            if self.handle.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_meta.append(item)
            else:
                self.out_meta.append(item)
        self._outputs: dict[tuple[Any, ...], Any] = {}
        self._input_references: list[Any] = []
        self.last_outputs: dict[str, Any] = {}
        self._stream_handle: int | None = None
        self._stream_device: int | None = None
        self._last_execution_event = torch.cuda.Event(blocking=False)
        self.load_count = 1
        self.context_count = 1
        self.allocation_count = 0
        self.execute_count = 0
        self.event_record_count = 0
        self.close_event_sync_count = 0
        self.closed = False

    def _torch_type(self, dtype: Any) -> Any:
        trt = self._trt
        torch = self._torch
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
            trt.uint8: torch.uint8,
            trt.int64: torch.int64,
        }
        try:
            return mapping[dtype]
        except KeyError as error:
            raise TypeError(f"unsupported TensorRT dtype: {dtype}") from error

    def dtype_of(self, tensor_name: str) -> Any:
        for name, _shape, dtype in self.in_meta:
            if name == tensor_name:
                return dtype
        raise KeyError(f"TensorRT input does not exist: {tensor_name}")

    def set_runtime_tensor_shape(self, name: str, shape: Any) -> None:
        if not self.execution_context.set_input_shape(name, tuple(shape)):
            raise ValueError(
                f"TensorRT rejected runtime shape for {name}: {tuple(shape)}"
            )

    def _output(self, name: str, dtype: Any, device: Any) -> Any:
        shape = tuple(self.execution_context.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(
                f"TensorRT output shape remains unresolved for {name}: {shape}"
            )
        key = (name, shape, dtype, device.index)
        if key not in self._outputs:
            self._outputs[key] = self._torch.empty(shape, dtype=dtype, device=device)
            self.allocation_count += 1
        return self._outputs[key]

    def __call__(self, *args: Any, **inputs: Any) -> Any:
        return_list = bool(inputs.pop("return_list", False))
        if self.closed:
            raise RuntimeError("TensorRT engine is closed")
        if len(args) > len(self.in_meta):
            raise ValueError("too many positional TensorRT inputs")
        values = {self.in_meta[index][0]: value for index, value in enumerate(args)}
        duplicate = set(values) & set(inputs)
        if duplicate:
            raise ValueError(f"duplicate TensorRT inputs: {sorted(duplicate)}")
        values.update(inputs)
        expected = {item[0] for item in self.in_meta}
        if set(values) != expected:
            raise ValueError(
                f"TensorRT input mismatch: missing={sorted(expected - set(values))}, "
                f"extra={sorted(set(values) - expected)}"
            )

        references = []
        device = None
        for name, _shape, dtype in self.in_meta:
            value = values[name]
            if not isinstance(value, self._torch.Tensor) or not value.is_cuda:
                raise TypeError(f"TensorRT input {name} must be a CUDA tensor")
            if value.dtype != dtype:
                raise TypeError(
                    f"TensorRT input {name} dtype {value.dtype} does not match {dtype}"
                )
            if not value.is_contiguous():
                raise ValueError(
                    f"TensorRT input {name} must be contiguous; implicit copies are "
                    "forbidden in the persistent path"
                )
            runtime_shape = tuple(self.execution_context.get_tensor_shape(name))
            if runtime_shape != tuple(value.shape):
                raise ValueError(
                    f"TensorRT input {name} shape {tuple(value.shape)} does not match "
                    f"the selected runtime shape {runtime_shape}"
                )
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError("all TensorRT inputs must be on one CUDA device")
            self.execution_context.set_tensor_address(name, value.data_ptr())
            references.append(value)
        if device is None:
            raise RuntimeError("TensorRT engine has no inputs")

        outputs = {}
        for name, _shape, dtype in self.out_meta:
            value = self._output(name, dtype, device)
            self.execution_context.set_tensor_address(name, value.data_ptr())
            outputs[name] = value
        stream = self._torch.cuda.current_stream(device)
        stream_handle = int(stream.cuda_stream)
        if self._stream_handle is None:
            self._stream_handle = stream_handle
            self._stream_device = device.index
        elif (
            stream_handle != self._stream_handle or device.index != self._stream_device
        ):
            raise RuntimeError(
                "PersistentEngine is single-stream: cross-stream output reuse is "
                "forbidden without an explicit event handoff"
            )
        if not self.execution_context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed: {self.file}")
        self._last_execution_event.record(stream)
        self.event_record_count += 1
        self.execute_count += 1
        self._input_references = references
        self.last_outputs = outputs
        if return_list:
            return [outputs[item[0]] for item in self.out_meta]
        return outputs

    def telemetry(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "load_count": self.load_count,
            "context_count": self.context_count,
            "allocation_count": self.allocation_count,
            "execute_count": self.execute_count,
            "submission_api": "execute_async_v3",
            "stream_protocol": "single_torch_current_stream",
            "stream_handle": self._stream_handle,
            "stream_device": self._stream_device,
            "event_record_count": self.event_record_count,
            "resident_host_sync_count": 0,
            "close_event_sync_count": self.close_event_sync_count,
            "output_cache_entries": len(self._outputs),
            "closed": self.closed,
        }

    def close(self) -> None:
        if self.closed:
            return
        if self.event_record_count:
            self._last_execution_event.synchronize()
            self.close_event_sync_count += 1
        self.closed = True
        self._input_references.clear()
        self.last_outputs.clear()
        self._outputs.clear()
        self._last_execution_event = None
        self.execution_context = None
        self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
