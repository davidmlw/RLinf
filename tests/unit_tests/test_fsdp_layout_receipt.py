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
from types import SimpleNamespace

import torch
import torch.nn as nn

from rlinf.hybrid_engines.fsdp.layout_receipt import (
    describe_fsdp_layout,
    write_fsdp_layout_receipt_from_env,
)


def _fake_layout():
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    parameter = model[0].weight
    flat_param = torch.empty(4, dtype=torch.bfloat16)
    flat_param._unpadded_unsharded_size = torch.Size([16])
    flat_param._param_infos = [
        SimpleNamespace(module=model[0], module_name="0", param_name="weight")
    ]
    flat_param._numels = [16]
    flat_param._params = [parameter]
    handle = SimpleNamespace(flat_param=flat_param, uses_sharded_strategy=True)
    model._handle = handle
    return model, [handle]


def test_describe_fsdp_layout_reports_full_and_local_sizes():
    model, handles = _fake_layout()

    value = describe_fsdp_layout(model, handles, world_size=4)

    assert value["handle_count"] == 1
    assert value["total_full_numel"] == 16
    entry = value["handles"][0]
    assert entry["module"] == "<root>"
    assert entry["dtype"] == "bfloat16"
    assert entry["local_numel"] == 4
    assert entry["full_parameter_bytes"] == 32
    assert entry["largest_original_parameters"] == [
        {"fqn": "0.weight", "numel": 16, "requires_grad": True}
    ]


def test_layout_writer_is_disabled_without_environment(monkeypatch):
    model, handles = _fake_layout()
    monkeypatch.delenv("RLINF_FSDP_LAYOUT_DIR", raising=False)

    assert write_fsdp_layout_receipt_from_env(model, handles, 4) is None


def test_layout_writer_uses_rank_specific_path(tmp_path, monkeypatch):
    model, handles = _fake_layout()
    monkeypatch.setenv("RLINF_FSDP_LAYOUT_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")

    path = write_fsdp_layout_receipt_from_env(model, handles, 4)

    assert path == tmp_path / "fsdp-layout-rank-3.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["rank"] == 3
    assert value["world_size"] == 4
