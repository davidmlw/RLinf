# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Freeze W85 algorithm defaults while admitting the explicit RTX runtime."""

import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]


class ReferenceContractTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(
            "os.environ", {"EMBODIED_PATH": str(ROOT / "examples/embodiment")}
        ):
            with initialize_config_dir(
                config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
            ):
                self.original = compose(
                    config_name="isaaclab_franka_stack_cube_ppo_gr00t"
                )
                self.control = compose(
                    config_name="isaaclab_franka_stack_cube_ppo_gr00t_rtx_reference"
                )

    def test_algorithm_optimizer_and_fsdp_remain_original(self):
        for key in ("algorithm", "actor.optim", "actor.fsdp_config"):
            self.assertEqual(
                OmegaConf.to_container(OmegaConf.select(self.original, key)),
                OmegaConf.to_container(OmegaConf.select(self.control, key)),
            )
        self.assertEqual(self.control.actor.model.rl_head_config.noise_level, 0.5)
        self.assertTrue(self.control.actor.fsdp_config.gradient_checkpointing)

    def test_workload_and_role_placement(self):
        c = self.control
        self.assertEqual(c.cluster.component_placement.env, "0-7")
        self.assertEqual(c.cluster.component_placement.actor, "0-7")
        self.assertEqual(c.cluster.component_placement.rollout.placement, "0-7")
        self.assertEqual(c.env.train.total_num_envs, 64)
        self.assertEqual(c.env.train.rollout_epoch, 2)
        self.assertEqual(c.env.train.max_steps_per_rollout_epoch, 448)
        self.assertEqual(c.actor.global_batch_size, 512)
        self.assertEqual(c.actor.micro_batch_size, 32)
        actions = 64 * 2 * 448
        records = actions // c.actor.model.num_action_chunks
        self.assertEqual(
            (actions, records, records // 512 * c.algorithm.update_epoch),
            (57344, 3584, 21),
        )

    def test_evaluation_schedule_and_bounded_retention(self):
        c = self.control
        self.assertEqual(c.runner.max_steps, 100)
        self.assertEqual(list(c.runner.evaluation_steps), [0, 20, 50, 100])
        self.assertEqual(list(c.runner.checkpoint_steps), [20, 40, 50, 60, 80, 100])
        self.assertEqual(c.runner.checkpoint_keep_last, 1)
        self.assertEqual(c.env.eval.total_num_envs * c.env.eval.rollout_epoch, 192)
        self.assertNotEqual(c.env.eval.seed, c.env.train.seed)
        self.assertNotIn("external_simulator", self.original.env.train)


if __name__ == "__main__":
    unittest.main()
