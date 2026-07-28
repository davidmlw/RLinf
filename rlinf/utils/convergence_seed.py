# Copyright 2025 The RLinf Authors.
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

"""W64 convergence instrumentation: deterministic rollout RNG seeding.

Semantically neutral. This module only fixes the *starting point* of the
already-existing global-RNG draws used by GR00T flow-matching action sampling
(``torch.randn`` / ``torch.normal`` without an explicit generator), so that a
paired convergence A/B is reproducible. It changes no transport, model,
optimizer, workload, reward, or policy semantics; it is applied byte-identically
to V0 and V1 so it cannot bias the comparison; and it is opt-in (no reseed
unless ``rollout.seed`` is set, preserving un-instrumented default behavior).

The single injected worker call is ``seed_rollout_step`` inside
``set_global_step``. Reproducibility of the rollout/trajectory is verified
transitively from per-step metrics and per-step checkpoint tensors, so no extra
worker hook is required.
"""

from __future__ import annotations

from typing import Any

import torch


def derive_rollout_step_seed(rollout_seed: int, rank: int, global_step: int) -> int:
    """Collision-free, human-auditable per-(seed, rank, step) rollout seed.

    seed / rank / step occupy disjoint digit bands for the 8-rank, <=100-step
    production workload (rank 0-7 < 8e6; step 0-99 < 1e6), and the result fits
    in int64.
    """
    return int(rollout_seed) * 10_000_000 + int(rank) * 1_000_000 + int(global_step)


def seed_rollout_step(worker: Any, global_step: int) -> None:
    """Reseed the rollout process global torch RNG for this training step.

    No-op unless ``rollout.seed`` is configured. Reads the seed from config each
    call (cheap) so no worker ``__init__`` change is needed, keeping the applied
    patch minimal and identical across V0/V1. Logs the derived seed so a smoke
    can verify it against the closed-form formula.
    """
    rollout_seed = worker.cfg.rollout.get("seed", None)
    if rollout_seed is None:
        return
    step_seed = derive_rollout_step_seed(rollout_seed, worker._rank, global_step)
    torch.manual_seed(step_seed)
    if worker.torch_platform.device_count() > 0:
        worker.torch_platform.manual_seed_all(step_seed)
    worker.log_info(
        f"W64_ROLLOUT_SEED rank={worker._rank} step={int(global_step)} "
        f"rollout_seed={int(rollout_seed)} step_seed={step_seed}"
    )
