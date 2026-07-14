# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]


def _method(path: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse((_REPO_ROOT / path).read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _call_lines(node: ast.AST, name: str) -> list[int]:
    return [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Name) and call.func.id == name)
            or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
        )
    ]


def test_optimizer_steps_retain_cuda_allocator_cache():
    actor_path = "rlinf/workers/actor/fsdp_actor_worker.py"
    methods = [
        _method(actor_path, "EmbodiedFSDPActor", "run_training"),
        _method(actor_path, "EmbodiedFSDPActor", "finish_global_batch"),
    ]

    for method in methods:
        assert not _call_lines(method, "empty_cache")


def test_training_paths_clear_memory_after_the_complete_update():
    actor_run = _method(
        "rlinf/workers/actor/fsdp_actor_worker.py",
        "EmbodiedFSDPActor",
        "run_training",
    )
    pipeline_run = _method(
        "rlinf/workers/actor/fsdp_actor_worker_pipeline.py",
        "PipelineEmbodiedFSDPActor",
        "run_training",
    )

    actor_cleanup = _call_lines(actor_run, "clear_memory")
    actor_steps = _call_lines(actor_run, "optimizer_step")
    assert len(actor_cleanup) == 1
    assert actor_steps
    assert actor_cleanup[0] > max(actor_steps)

    pipeline_cleanup = _call_lines(pipeline_run, "clear_memory")
    pipeline_steps = _call_lines(pipeline_run, "finish_global_batch")
    assert len(pipeline_cleanup) == 1
    assert pipeline_steps
    assert pipeline_cleanup[0] > max(pipeline_steps)
