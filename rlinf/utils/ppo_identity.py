# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Numerical receipts for pre-update behavior-policy identity checks."""

from __future__ import annotations

from collections.abc import Mapping

import torch

_DIAGNOSTIC_CUTOFFS = (1e-4, 1e-3, 1e-2)


def pre_update_identity_batch_stats(
    *,
    current_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    current_values: torch.Tensor,
    behavior_values: torch.Tensor,
    loss_mask: torch.Tensor | None,
    logprob_type: str,
    single_action_dim: int,
) -> dict[str, torch.Tensor]:
    """Return additive/max statistics without hiding non-finite values."""

    if current_logprobs.shape != behavior_logprobs.shape:
        raise ValueError(
            "current and behavior log-probability shapes differ: "
            f"{tuple(current_logprobs.shape)} != {tuple(behavior_logprobs.shape)}"
        )

    batch_size = current_logprobs.shape[0]
    if logprob_type == "token_level":
        current_logprobs = current_logprobs.reshape(
            batch_size, -1, single_action_dim
        )
        behavior_logprobs = behavior_logprobs.reshape(
            batch_size, -1, single_action_dim
        )
    elif logprob_type == "action_level":
        current_logprobs = current_logprobs.reshape(
            batch_size, -1, single_action_dim
        ).sum(dim=-1)
        behavior_logprobs = behavior_logprobs.reshape(
            batch_size, -1, single_action_dim
        ).sum(dim=-1)
    elif logprob_type == "chunk_level":
        current_logprobs = current_logprobs.reshape(
            batch_size, -1, single_action_dim
        ).sum(dim=(1, 2))
        behavior_logprobs = behavior_logprobs.reshape(
            batch_size, -1, single_action_dim
        ).sum(dim=(1, 2))
    else:
        raise ValueError(f"unsupported log-probability type: {logprob_type!r}")

    log_ratio = current_logprobs.float() - behavior_logprobs.float()
    if loss_mask is None:
        mask = torch.ones_like(log_ratio, dtype=torch.bool)
    else:
        mask = loss_mask.to(device=log_ratio.device, dtype=torch.bool)
        while mask.ndim < log_ratio.ndim:
            mask = mask.unsqueeze(-1)
        try:
            mask = mask.expand_as(log_ratio)
        except RuntimeError as error:
            raise ValueError(
                f"loss mask {tuple(mask.shape)} cannot cover log-probabilities "
                f"{tuple(log_ratio.shape)}"
            ) from error

    ratio_delta = torch.expm1(log_ratio)
    finite_ratio = torch.isfinite(log_ratio) & torch.isfinite(ratio_delta) & mask
    ratio_count = mask.count_nonzero()
    ratio_finite_count = finite_ratio.count_nonzero()
    ratio_abs = ratio_delta.abs()[finite_ratio]
    kl_abs = log_ratio.abs()[finite_ratio]

    current_values = current_values.float().reshape(-1)
    behavior_values = behavior_values.float().reshape(-1)
    if current_values.shape != behavior_values.shape:
        raise ValueError(
            "current and behavior value shapes differ after flattening: "
            f"{tuple(current_values.shape)} != {tuple(behavior_values.shape)}"
        )
    value_delta = (current_values - behavior_values).abs()
    finite_value = torch.isfinite(value_delta)
    value_count = torch.tensor(
        value_delta.numel(), device=value_delta.device, dtype=torch.int64
    )
    value_finite_count = finite_value.count_nonzero()
    finite_value_delta = value_delta[finite_value]

    zero = torch.zeros((), device=log_ratio.device, dtype=torch.float64)

    def _sum(value: torch.Tensor) -> torch.Tensor:
        return value.double().sum() if value.numel() else zero.clone()

    def _max(value: torch.Tensor) -> torch.Tensor:
        return value.double().max() if value.numel() else zero.clone()

    stats = {
        "decision_records": torch.tensor(
            batch_size, device=log_ratio.device, dtype=torch.int64
        ),
        "ratio_positions": ratio_count.to(dtype=torch.int64),
        "nonfinite_ratio_positions": (ratio_count - ratio_finite_count).to(
            dtype=torch.int64
        ),
        "ratio_abs_sum": _sum(ratio_abs),
        "ratio_abs_max": _max(ratio_abs),
        "kl_abs_sum": _sum(kl_abs),
        "kl_abs_max": _max(kl_abs),
        "value_positions": value_count,
        "nonfinite_value_positions": (value_count - value_finite_count).to(
            dtype=torch.int64
        ),
        "value_abs_sum": _sum(finite_value_delta),
        "value_abs_max": _max(finite_value_delta),
    }
    for cutoff in _DIAGNOSTIC_CUTOFFS:
        label = f"{cutoff:.0e}".replace("-0", "-")
        stats[f"ratio_abs_gt_{label}"] = (ratio_abs > cutoff).count_nonzero()
        stats[f"kl_abs_gt_{label}"] = (kl_abs > cutoff).count_nonzero()
    return stats


def finalize_pre_update_identity(
    totals: Mapping[str, int | float],
    thresholds: Mapping[str, float],
) -> dict[str, int | float | bool]:
    """Finalize reduced statistics and apply pre-registered thresholds."""

    ratio_positions = int(totals["ratio_positions"])
    value_positions = int(totals["value_positions"])
    if ratio_positions <= 0 or value_positions <= 0:
        raise ValueError("identity gate received no ratio or value positions")

    receipt: dict[str, int | float | bool] = {
        "decision_records": int(totals["decision_records"]),
        "ratio_positions": ratio_positions,
        "nonfinite_ratio_positions": int(totals["nonfinite_ratio_positions"]),
        "ratio_mean_abs_from_one": float(totals["ratio_abs_sum"]) / ratio_positions,
        "ratio_max_abs_from_one": float(totals["ratio_abs_max"]),
        "kl_mean_abs": float(totals["kl_abs_sum"]) / ratio_positions,
        "kl_max_abs": float(totals["kl_abs_max"]),
        "value_positions": value_positions,
        "nonfinite_value_positions": int(totals["nonfinite_value_positions"]),
        "value_mean_abs": float(totals["value_abs_sum"]) / value_positions,
        "value_max_abs": float(totals["value_abs_max"]),
    }
    for cutoff in _DIAGNOSTIC_CUTOFFS:
        label = f"{cutoff:.0e}".replace("-0", "-")
        for prefix in ("ratio_abs_gt", "kl_abs_gt"):
            key = f"{prefix}_{label}"
            count = int(totals[key])
            receipt[key] = count
            receipt[f"{key}_fraction"] = count / ratio_positions
    receipt["finite"] = (
        receipt["nonfinite_ratio_positions"] == 0
        and receipt["nonfinite_value_positions"] == 0
    )
    receipt["passed"] = bool(
        receipt["finite"]
        and receipt["ratio_mean_abs_from_one"]
        <= float(thresholds["ratio_mean_abs_from_one_max"])
        and receipt["ratio_max_abs_from_one"]
        <= float(thresholds["ratio_max_abs_from_one_max"])
        and receipt["kl_mean_abs"] <= float(thresholds["kl_mean_abs_max"])
        and receipt["kl_max_abs"] <= float(thresholds["kl_max_abs_max"])
    )
    return receipt
