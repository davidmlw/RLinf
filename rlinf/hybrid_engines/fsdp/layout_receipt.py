# Copyright 2025 The RLinf Authors.
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

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _numel(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "numel"):
        return int(value.numel())
    try:
        return math.prod(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def describe_fsdp_layout(model: nn.Module, handles: list, world_size: int) -> dict:
    module_names = {
        id(module): name or "<root>" for name, module in model.named_modules()
    }
    handle_modules: dict[int, str] = {}
    for name, module in model.named_modules():
        handle = getattr(module, "_handle", None)
        if handle is not None:
            handle_modules[id(handle)] = name or "<root>"

    entries = []
    for index, handle in enumerate(handles):
        flat_param = handle.flat_param
        local_numel = int(flat_param.numel())
        full_numel = _numel(
            getattr(flat_param, "_unpadded_unsharded_size", None)
        )
        if full_numel is None:
            full_numel = local_numel * world_size

        param_infos = list(getattr(flat_param, "_param_infos", ()))
        param_numels = list(getattr(flat_param, "_numels", ()))
        params = list(getattr(flat_param, "_params", ()))
        parameters = []
        trainable_numel = 0
        frozen_numel = 0
        unknown_numel = 0
        for param_index, (info, param_numel) in enumerate(
            zip(param_infos, param_numels)
        ):
            module = getattr(info, "module", None)
            module_name = module_names.get(id(module))
            if module_name is None:
                module_name = getattr(info, "module_name", "<unknown>") or "<root>"
            param_name = getattr(info, "param_name", f"parameter_{param_index}")
            fqn = (
                param_name
                if module_name == "<root>"
                else f"{module_name}.{param_name}"
            )
            requires_grad = None
            if param_index < len(params):
                requires_grad = bool(params[param_index].requires_grad)
            if requires_grad is True:
                trainable_numel += int(param_numel)
            elif requires_grad is False:
                frozen_numel += int(param_numel)
            else:
                unknown_numel += int(param_numel)
            parameters.append(
                {
                    "fqn": fqn,
                    "numel": int(param_numel),
                    "requires_grad": requires_grad,
                }
            )

        parameters.sort(key=lambda value: (-value["numel"], value["fqn"]))
        element_size = int(flat_param.element_size())
        entries.append(
            {
                "index": index,
                "module": handle_modules.get(id(handle), "<unknown>"),
                "dtype": str(flat_param.dtype).removeprefix("torch."),
                "uses_sharded_strategy": bool(handle.uses_sharded_strategy),
                "local_numel": local_numel,
                "full_numel": full_numel,
                "full_parameter_bytes": full_numel * element_size,
                "original_parameter_count": len(param_infos),
                "trainable_parameter_bytes": trainable_numel * element_size,
                "frozen_parameter_bytes": frozen_numel * element_size,
                "unknown_parameter_bytes": unknown_numel * element_size,
                "largest_original_parameters": parameters[:16],
            }
        )

    return {
        "schema": "rlinf.fsdp-layout/v1",
        "world_size": world_size,
        "handle_count": len(entries),
        "total_full_numel": sum(entry["full_numel"] for entry in entries),
        "total_full_parameter_bytes": sum(
            entry["full_parameter_bytes"] for entry in entries
        ),
        "handles": entries,
    }


def write_fsdp_layout_receipt_from_env(
    model: nn.Module, handles: list, world_size: int
) -> Path | None:
    output_dir = os.environ.get("RLINF_FSDP_LAYOUT_DIR")
    if not output_dir:
        return None

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"fsdp-layout-rank-{rank}.json"
    temporary = root / f".{path.name}.tmp-{os.getpid()}"
    value = describe_fsdp_layout(model, handles, world_size)
    value["rank"] = rank
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path
