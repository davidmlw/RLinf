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

"""Deterministic rollout and critic initialization for paired experiments.

Semantically neutral. This module fixes only the *starting point* of
already-existing global-RNG draws so that a paired convergence A/B is
reproducible. It changes no transport, model, optimizer, workload, reward, or
policy semantics; it is applied byte-identically to V0 and V1 so it cannot bias
the comparison; and every piece is opt-in (default-off preserves the
un-instrumented behavior).

Two instrumentation points, both opt-in:

1. Rollout RNG (``seed_rollout_step`` inside ``set_global_step``): fixes the
   global-RNG flow-matching action noise per (rollout_seed, rank, step). No-op
   unless ``rollout.seed`` is set.

2. Critic-init RNG (``maybe_seeded_value_head_init`` at model build): the PPO
   value/critic head is absent from the pretrained SFT checkpoint and is
   randomly re-initialized from the *global* RNG at build; unseeded it differs
   every run/arm and confounds the A/B. No-op unless
   ``actor.model.value_head_init_seed`` is set; when set it seeds an isolated
   CPU RNG scope (global RNG restored on exit).

Reproducibility of the rollout/trajectory is verified transitively from
per-step metrics and per-step checkpoint tensors, so no extra worker hook is
required.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch


def _log_seed_event(message: str) -> None:
    """Emit an instrumentation line that is guaranteed to reach the run log.

    Inside a rlinf Worker (e.g. the Actor model build) the raw ``print`` FD is
    NOT forwarded to the driver/bench log, but the worker logger IS; outside a
    Worker (unit tests) there is no worker logger. Try the worker logger first
    and fall back to a flushed print so both contexts surface the line.
    """
    try:
        from rlinf.utils.logging import get_logger

        get_logger().warning(message)
    except Exception:
        print(message, flush=True)


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
        f"RLINF_ROLLOUT_SEED rank={worker._rank} step={int(global_step)} "
        f"rollout_seed={int(rollout_seed)} step_seed={step_seed}"
    )


def critic_digest(value_head: Any) -> str:
    """Order-stable sha256 over the critic head weight *values* (not an archive
    hash), for proving V0 and V1 (and run-a vs run-b) start PPO from an
    identical critic. Dtype-agnostic: reinterprets each tensor's bytes via a
    uint8 view so bf16 params (which have no numpy dtype) hash cleanly.
    """
    h = hashlib.sha256()
    for name, param in sorted(value_head.state_dict().items()):
        h.update(name.encode("utf-8"))
        flat = param.detach().cpu().contiguous().flatten().view(torch.uint8)
        h.update(flat.numpy().tobytes())
    return h.hexdigest()


def maybe_seeded_value_head_init(value_head: Any, init_seed: Any) -> None:
    """Deterministically (re)initialize the PPO value/critic head.

    ``value_head._init_weights`` draws from the *global* torch RNG, so left
    unseeded every run/arm gets a different critic, which confounds a paired
    convergence A/B (the critic is not in the pretrained checkpoint). When
    ``init_seed`` is set, seed a *local* CPU RNG scope so the critic starts
    identically for a given seed WITHOUT perturbing the global RNG stream used
    elsewhere; when it is ``None`` the behavior is exactly the un-instrumented
    default (bare ``_init_weights``).

    Both branches log ``RLINF_CRITIC_INIT`` (seeded vs skipped + the seed value
    actually seen) so a run can prove which path executed. Applied
    byte-identically to V0 and V1, and with NO rank offset: the critic is a
    data-parallel parameter and FSDP ``sync_module_states=True`` broadcasts
    rank 0, so all Actor ranks must start identical within a run. The CPU
    assertion makes the ``fork_rng(devices=[])`` isolation load-bearing: a
    non-CPU init would escape the fork and is turned into an error, not silent
    global-RNG pollution.
    """
    if init_seed is None:
        _log_seed_event(
            "RLINF_CRITIC_INIT skipped: value_head_init_seed=None (unseeded default)"
        )
        value_head._init_weights()
        return
    devices = {p.device.type for p in value_head.parameters()}
    assert devices <= {"cpu"}, (
        f"seeded critic init expects CPU parameters, got {sorted(devices)}; "
        "fork_rng(devices=[]) would not isolate a non-CPU RNG"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        value_head._init_weights()
    _log_seed_event(
        f"RLINF_CRITIC_INIT seed={int(init_seed)} devices={sorted(devices)} "
        f"digest={critic_digest(value_head)}"
    )
