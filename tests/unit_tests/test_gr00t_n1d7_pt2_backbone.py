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

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PT2_SOURCE = (
    ROOT / "rlinf/models/embodiment/gr00t/gr00t_n1d7/pt2_backbone.py"
).read_text(encoding="utf-8")


def _load_geometry_validator() -> Callable:
    tree = ast.parse(PT2_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_uniform_vision_sequence_length"
    )
    namespace = {}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="collections.abc",
                names=[ast.alias(name="Sequence")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    code = compile(ast.fix_missing_locations(module), "pt2_backbone.py", "exec")
    exec(code, namespace)
    return namespace["_uniform_vision_sequence_length"]


_uniform_vision_sequence_length = _load_geometry_validator()


def test_uniform_true_b8_trocar_geometry() -> None:
    assert _uniform_vision_sequence_length([[1, 16, 16]] * 24) == (256, 24)


def test_static_geometry_rejects_mixed_camera_grid() -> None:
    with pytest.raises(ValueError, match="one sequence length"):
        _uniform_vision_sequence_length([[1, 16, 16], [1, 14, 14]])


def test_rollout_worker_rejects_two_backbone_executors() -> None:
    source = (ROOT / "rlinf/workers/rollout/hf/huggingface_worker.py").read_text(
        encoding="utf-8"
    )
    assert "pt2_backbone_enabled and tensorrt_backbone_enabled" in source
    assert "configure either rollout.model.torch_compile_backbone or" in source
    assert "rollout.model.tensorrt_backbone, not both" in source


def test_model_routes_pt2_before_eager_backbone() -> None:
    source = (
        ROOT
        / "rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_backbone"
    )
    names = [
        node.args[1].value
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
    ]
    assert names[:2] == ["_tensorrt_backbone", "_pt2_backbone"]
