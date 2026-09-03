# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from rlinf.utils.ppo_identity import (
    finalize_pre_update_identity,
    pre_update_identity_batch_stats,
)

THRESHOLDS = {
    "ratio_mean_abs_from_one_max": 1e-4,
    "ratio_max_abs_from_one_max": 1e-3,
    "kl_mean_abs_max": 1e-4,
    "kl_max_abs_max": 1e-3,
}


def _as_scalars(stats: dict[str, torch.Tensor]) -> dict[str, int | float]:
    return {key: value.item() for key, value in stats.items()}


def test_identity_stats_expand_chunk_mask_and_pass_exact_policy() -> None:
    behavior = torch.tensor([[[0.1, -0.2], [0.3, -0.4]]])
    stats = pre_update_identity_batch_stats(
        current_logprobs=behavior.clone(),
        behavior_logprobs=behavior,
        current_values=torch.tensor([[0.5]]),
        behavior_values=torch.tensor([0.5]),
        loss_mask=torch.tensor([[[True], [False]]]),
        logprob_type="token_level",
        single_action_dim=2,
    )
    receipt = finalize_pre_update_identity(_as_scalars(stats), THRESHOLDS)

    assert receipt["decision_records"] == 1
    assert receipt["ratio_positions"] == 2
    assert receipt["ratio_mean_abs_from_one"] == 0.0
    assert receipt["kl_max_abs"] == 0.0
    assert receipt["value_mean_abs"] == 0.0
    assert receipt["passed"] is True


def test_identity_stats_fail_closed_on_threshold_and_nonfinite() -> None:
    behavior = torch.zeros((1, 1, 2))
    current = torch.tensor([[[2e-3, float("nan")]]])
    stats = pre_update_identity_batch_stats(
        current_logprobs=current,
        behavior_logprobs=behavior,
        current_values=torch.tensor([0.7]),
        behavior_values=torch.tensor([0.5]),
        loss_mask=None,
        logprob_type="token_level",
        single_action_dim=2,
    )
    receipt = finalize_pre_update_identity(_as_scalars(stats), THRESHOLDS)

    assert receipt["nonfinite_ratio_positions"] == 1
    assert receipt["ratio_max_abs_from_one"] > 1e-3
    assert receipt["value_mean_abs"] == pytest.approx(0.2)
    assert receipt["finite"] is False
    assert receipt["passed"] is False


def test_identity_stats_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="log-probability shapes differ"):
        pre_update_identity_batch_stats(
            current_logprobs=torch.zeros((2, 3)),
            behavior_logprobs=torch.zeros((2, 4)),
            current_values=torch.zeros(2),
            behavior_values=torch.zeros(2),
            loss_mask=None,
            logprob_type="action_level",
            single_action_dim=2,
        )


def test_identity_stats_use_ppo_action_level_reduction() -> None:
    behavior = torch.zeros((1, 2, 2))
    current = torch.tensor([[[6e-4, 6e-4], [4e-4, -4e-4]]])
    stats = pre_update_identity_batch_stats(
        current_logprobs=current,
        behavior_logprobs=behavior,
        current_values=torch.zeros(1),
        behavior_values=torch.zeros(1),
        loss_mask=torch.ones((1, 2), dtype=torch.bool),
        logprob_type="action_level",
        single_action_dim=2,
    )
    receipt = finalize_pre_update_identity(_as_scalars(stats), THRESHOLDS)

    assert receipt["decision_records"] == 1
    assert receipt["ratio_positions"] == 2
    assert receipt["ratio_max_abs_from_one"] == pytest.approx(
        torch.expm1(torch.tensor(1.2e-3)).item()
    )
    assert receipt["passed"] is False
