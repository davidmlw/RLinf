# ruff: noqa: E402

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rlinf.utils.convergence_seed import (
    derive_rollout_step_seed,
    maybe_seeded_value_head_init,
    seed_rollout_step,
)


class _ValueHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(8, 8))

    def _init_weights(self) -> None:
        torch.nn.init.normal_(self.weight)


class _Platform:
    def __init__(self) -> None:
        self.seed = None

    def device_count(self) -> int:
        return 1

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed


class _Worker:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(rollout={"seed": 64101})
        self._rank = 3
        self.torch_platform = _Platform()
        self.messages = []

    def log_info(self, message: str) -> None:
        self.messages.append(message)


def test_rollout_seed_formula_has_no_rank_step_collisions() -> None:
    seeds = {
        derive_rollout_step_seed(64101, rank, step)
        for rank in range(8)
        for step in range(1000)
    }

    assert len(seeds) == 8000


def test_rollout_step_seeds_cpu_and_platform_rng() -> None:
    worker = _Worker()

    seed_rollout_step(worker, 7)

    expected = derive_rollout_step_seed(64101, 3, 7)
    assert worker.torch_platform.seed == expected
    assert f"step_seed={expected}" in worker.messages[0]


def test_seeded_critic_init_is_repeatable_and_restores_global_rng() -> None:
    torch.manual_seed(17)
    expected_next = torch.rand(4)
    torch.manual_seed(17)
    first = _ValueHead()
    maybe_seeded_value_head_init(first, 64001)
    actual_next = torch.rand(4)
    second = _ValueHead()
    maybe_seeded_value_head_init(second, 64001)

    assert torch.equal(actual_next, expected_next)
    assert torch.equal(first.weight, second.weight)


def test_different_critic_seeds_produce_different_weights() -> None:
    first = _ValueHead()
    second = _ValueHead()

    maybe_seeded_value_head_init(first, 64001)
    maybe_seeded_value_head_init(second, 64002)

    assert not torch.equal(first.weight, second.weight)
