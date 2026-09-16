# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from toolkits.horde.gr00t_trocar.rollout_env_benchmark import (  # noqa: E402
    _statistics,
    actions_to_env,
    observation_to_policy,
)


def test_observation_to_policy_maps_three_cameras_and_four_states() -> None:
    observation = {
        "camera_images": {
            "front_camera": torch.full((8, 224, 224, 3), 1, dtype=torch.uint8),
            "left_wrist_camera": torch.full(
                (8, 224, 224, 3), 2, dtype=torch.uint8
            ),
            "right_wrist_camera": torch.full(
                (8, 224, 224, 3), 3, dtype=torch.uint8
            ),
        },
        "policy": {
            "robot_joint_state": torch.arange(8 * 87).reshape(8, 87).float(),
            "robot_dex3_joint_state": torch.arange(8 * 14).reshape(8, 14).float(),
        },
    }

    result = observation_to_policy(observation)

    assert tuple(result["video"]) == (
        "left_wrist_view",
        "right_wrist_view",
        "room_view",
    )
    assert result["video"]["room_view"].shape == (8, 1, 224, 224, 3)
    assert result["video"]["left_wrist_view"][0, 0, 0, 0, 0] == 2
    assert result["state"]["left_arm"].shape == (8, 1, 7)
    np.testing.assert_array_equal(
        result["state"]["left_arm"][0, 0], np.arange(15, 22, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result["state"]["right_hand"][0, 0], np.arange(7, 14, dtype=np.float32)
    )
    assert result["language"]["annotation.human.action.task_description"] == [
        ["assemble trocar from tray"]
    ] * 8


def test_actions_to_env_preserves_order_and_adds_prefix() -> None:
    actions = {
        f"action.{name}": np.full((8, 16, 7), index, dtype=np.float32)
        for index, name in enumerate(
            ("left_arm", "right_arm", "left_hand", "right_hand"), start=1
        )
    }

    result = actions_to_env(actions, torch.device("cpu"))

    assert tuple(result.shape) == (8, 16, 43)
    assert torch.count_nonzero(result[..., :15]) == 0
    for index in range(4):
        assert torch.all(result[..., 15 + index * 7 : 22 + index * 7] == index + 1)


def test_statistics_retains_raw_samples() -> None:
    result = _statistics([1.0, 2.0, 3.0])

    assert result["count"] == 3
    assert result["samples_ms"] == [1.0, 2.0, 3.0]
    assert result["mean_ms"] == 2.0
    assert result["p50_ms"] == 2.0
