# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf extension module for IsaacLab tasks.

This module is loaded by RLinf's Worker._load_user_extensions() when
RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension is set in the environment.

It registers IsaacLab tasks into RLinf's registries, allowing IsaacLab users
to train on their tasks without modifying RLinf source code.

Configuration is read from the Hydra YAML config under `env.train.isaaclab`:
    env:
      train:
        isaaclab: &isaaclab_config  # YAML anchor for reuse
          task_description: "..."
          main_images: "front_camera"
          extra_view_images: ["left_wrist_camera", "right_wrist_camera"]
          states:
            - key: "robot_joint_state"
              slice: [15, 29]
          gr00t_mapping:
            video:
              main_images: "video.room_view"
              ...
          action_mapping:
            prefix_pad: 15
      eval:
        isaaclab: *isaaclab_config  # Reuse via YAML anchor

Task IDs are read automatically from ``env.train.init_params.id`` and
``env.eval.init_params.id`` in the YAML config.

Usage:
    export RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension
    export RLINF_CONFIG_FILE=/path/to/isaaclab_ppo_gr00t_assemble_trocar.yaml
"""

from __future__ import annotations

import collections.abc
import json
import logging
import os
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

from rlinf.models.embodiment.gr00t import embodiment_tags

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_registered = False

# Cache for YAML config (loaded once per process)
_full_cfg_cache: dict | None = None


def _assert_finite(value: object, *, stage: str) -> None:
    """Fail with a narrow diagnostic when a floating tensor becomes non-finite."""
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.floating):
            return
        finite = np.isfinite(value)
        if bool(finite.all()):
            return
        finite_values = value[finite]
        first_index = tuple(int(item) for item in np.argwhere(~finite)[0])
        value_min = float(finite_values.min()) if finite_values.size else None
        value_max = float(finite_values.max()) if finite_values.size else None
        raise RuntimeError(
            "RLINF_NONFINITE "
            f"stage={stage} shape={value.shape} dtype={value.dtype} "
            f"nonfinite={int((~finite).sum())} first_index={first_index} "
            f"finite_min={value_min} finite_max={value_max}"
        )
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        return
    finite = torch.isfinite(value)
    if bool(finite.all().item()):
        return
    nonfinite = int((~finite).sum().item())
    first_index = tuple(int(item) for item in torch.nonzero(~finite)[0].tolist())
    finite_values = value[finite]
    value_min = float(finite_values.min().item()) if finite_values.numel() else None
    value_max = float(finite_values.max().item()) if finite_values.numel() else None
    raise RuntimeError(
        "RLINF_NONFINITE "
        f"stage={stage} shape={tuple(value.shape)} dtype={value.dtype} "
        f"nonfinite={nonfinite} first_index={first_index} "
        f"finite_min={value_min} finite_max={value_max}"
    )


def _assert_magnitude(value: object, *, stage: str, limit: float) -> None:
    """Fail before a finite simulator value overflows during bf16 conversion."""
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.floating):
            return
        over_limit = np.abs(value) > limit
        if not bool(over_limit.any()):
            return
        first_index = tuple(int(item) for item in np.argwhere(over_limit)[0])
        raise RuntimeError(
            "RLINF_OUT_OF_RANGE "
            f"stage={stage} shape={value.shape} dtype={value.dtype} "
            f"limit={limit} count={int(over_limit.sum())} "
            f"first_index={first_index} first_value={float(value[first_index])} "
            f"value_min={float(value.min())} value_max={float(value.max())}"
        )
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        return
    over_limit = value.abs() > limit
    if not bool(over_limit.any().item()):
        return
    first_index = tuple(int(item) for item in torch.nonzero(over_limit)[0].tolist())
    raise RuntimeError(
        "RLINF_OUT_OF_RANGE "
        f"stage={stage} shape={tuple(value.shape)} dtype={value.dtype} "
        f"limit={limit} count={int(over_limit.sum().item())} "
        f"first_index={first_index} first_value={float(value[first_index].item())} "
        f"value_min={float(value.min().item())} value_max={float(value.max().item())}"
    )


def _max_abs(value: object) -> float | None:
    """Return a scalar magnitude for low-rate diagnostic telemetry."""
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.floating) or value.size == 0:
            return None
        return float(np.abs(value).max())
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point() or value.numel() == 0:
            return None
        return float(value.detach().abs().max().item())
    return None


def _update_env_peak(
    env: object, *, kind: str, value: object, physical_step: int
) -> None:
    peak = _max_abs(value)
    if peak is None:
        return
    value_attr = f"_rlinf_nonfinite_{kind}_max_abs"
    if peak > getattr(env, value_attr, -1.0):
        setattr(env, value_attr, peak)
        setattr(env, f"_rlinf_nonfinite_{kind}_peak_step", physical_step)


def _log_env_epoch_range(env: object) -> None:
    reset_count = getattr(env, "_rlinf_nonfinite_reset_count", 0)
    if reset_count == 0:
        return
    logger.warning(
        "RLINF_ENV_RANGE reset=%d physical_steps=%d "
        "action_max_abs=%s action_peak_step=%s "
        "state_max_abs=%s state_peak_step=%s",
        reset_count,
        getattr(env, "_rlinf_nonfinite_physical_step", 0),
        getattr(env, "_rlinf_nonfinite_action_max_abs", None),
        getattr(env, "_rlinf_nonfinite_action_peak_step", None),
        getattr(env, "_rlinf_nonfinite_state_max_abs", None),
        getattr(env, "_rlinf_nonfinite_state_peak_step", None),
    )


def _assert_nested_finite(value: object, *, stage: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_nested_finite(child, stage=f"{stage}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_nested_finite(child, stage=f"{stage}[{index}]")
    else:
        _assert_finite(value, stage=stage)


def _install_nonfinite_diagnostics() -> None:
    """Instrument rollout, value, and Env boundaries for W73 diagnostics."""
    try:
        from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
            FlowMatchingActionHeadForRLActionPrediction,
        )
    except ModuleNotFoundError:
        from rlinf.models.embodiment.gr00t.gr00t_action_model import (
            FlowMatchingActionHeadForRLActionPrediction,
        )
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    if getattr(
        FlowMatchingActionHeadForRLActionPrediction.get_value,
        "_rlinf_nonfinite_diagnostic",
        False,
    ):
        return

    original_get_value = FlowMatchingActionHeadForRLActionPrediction.get_value
    original_get_rl_action = (
        FlowMatchingActionHeadForRLActionPrediction.get_rl_action
    )

    def checked_get_rl_action(self, backbone_output, action_input, *args, **kwargs):
        _assert_nested_finite(
            backbone_output, stage="rollout.action_head.backbone_input"
        )
        _assert_nested_finite(action_input, stage="rollout.action_head.input")
        return original_get_rl_action(
            self, backbone_output, action_input, *args, **kwargs
        )

    def checked_get_value(self, vl_embs, state_features):
        _assert_finite(vl_embs, stage="value.backbone_features")
        _assert_finite(state_features, stage="value.state_features")
        values = original_get_value(self, vl_embs, state_features)
        _assert_finite(values, stage="value.output")
        return values

    checked_get_value._rlinf_nonfinite_diagnostic = True
    FlowMatchingActionHeadForRLActionPrediction.get_rl_action = checked_get_rl_action
    FlowMatchingActionHeadForRLActionPrediction.get_value = checked_get_value

    original_predict = MultiStepRolloutWorker._predict_rollout_actions

    def checked_predict(self, env_obs, *args, **kwargs):
        _assert_nested_finite(env_obs, stage="rollout.observation")
        _assert_magnitude(
            env_obs.get("states"),
            stage="rollout.observation.states",
            limit=100.0,
        )
        actions, result = original_predict(self, env_obs, *args, **kwargs)
        _assert_finite(actions, stage="rollout.actions")
        for key in ("prev_logprobs", "prev_values"):
            _assert_finite(result.get(key), stage=f"rollout.{key}")
        return actions, result

    original_sync = MultiStepRolloutWorker.sync_model_from_actor

    async def checked_sync(self):
        await original_sync(self)
        action_head = getattr(self.hf_model, "action_head", None)
        value_head = getattr(action_head, "value_head", None)
        if value_head is None:
            raise RuntimeError("RLINF_NONFINITE diagnostic could not find value_head")
        for name, parameter in value_head.named_parameters():
            _assert_finite(parameter, stage=f"rollout.post_sync.value_head.{name}")

    MultiStepRolloutWorker._predict_rollout_actions = checked_predict
    MultiStepRolloutWorker.sync_model_from_actor = checked_sync

    original_recv = EmbodiedFSDPActor.recv_rollout_trajectories

    async def checked_recv(self, input_channel):
        await original_recv(self, input_channel)
        _assert_nested_finite(self.rollout_batch, stage="actor.trajectory")

    original_compute_adv = EmbodiedFSDPActor.compute_advantages_and_returns

    def checked_compute_adv(self):
        for key in ("rewards", "prev_values", "prev_logprobs", "loss_mask"):
            _assert_finite(
                self.rollout_batch.get(key), stage=f"actor.gae_input.{key}"
            )
        metrics = original_compute_adv(self)
        for key in ("advantages", "returns"):
            _assert_finite(
                self.rollout_batch.get(key), stage=f"actor.gae_output.{key}"
            )
        return metrics

    EmbodiedFSDPActor.recv_rollout_trajectories = checked_recv
    EmbodiedFSDPActor.compute_advantages_and_returns = checked_compute_adv

    original_env_reset = IsaaclabBaseEnv.reset
    original_env_step = IsaaclabBaseEnv.step

    def checked_env_reset(self, *args, **kwargs):
        _log_env_epoch_range(self)
        obs, infos = original_env_reset(self, *args, **kwargs)
        self._rlinf_nonfinite_reset_count = (
            getattr(self, "_rlinf_nonfinite_reset_count", 0) + 1
        )
        self._rlinf_nonfinite_physical_step = 0
        self._rlinf_nonfinite_action_max_abs = -1.0
        self._rlinf_nonfinite_action_peak_step = None
        self._rlinf_nonfinite_state_max_abs = -1.0
        self._rlinf_nonfinite_state_peak_step = None
        stage = f"env.reset.{self._rlinf_nonfinite_reset_count}.states"
        _assert_finite(obs.get("states"), stage=stage)
        _assert_magnitude(obs.get("states"), stage=stage, limit=100.0)
        _update_env_peak(self, kind="state", value=obs.get("states"), physical_step=0)
        return obs, infos

    def checked_env_step(self, actions=None, *args, **kwargs):
        reset_count = getattr(self, "_rlinf_nonfinite_reset_count", 0)
        physical_step = getattr(self, "_rlinf_nonfinite_physical_step", 0) + 1
        self._rlinf_nonfinite_physical_step = physical_step
        action_stage = f"env.action.reset_{reset_count}.physical_{physical_step}"
        _assert_finite(actions, stage=action_stage)
        _assert_magnitude(actions, stage=action_stage, limit=100.0)
        _update_env_peak(
            self, kind="action", value=actions, physical_step=physical_step
        )
        result = original_env_step(self, actions, *args, **kwargs)
        obs, rewards, terminations, truncations, _infos = result
        state_stage = (
            f"env.step.output.reset_{reset_count}.physical_{physical_step}.states"
        )
        try:
            _assert_finite(obs.get("states"), stage=state_stage)
            _assert_magnitude(obs.get("states"), stage=state_stage, limit=100.0)
        except RuntimeError:
            logger.error(
                "RLINF_ENV_CONTEXT reset=%d physical_step=%d "
                "current_action_max_abs=%s epoch_action_max_abs=%s "
                "epoch_action_peak_step=%s",
                reset_count,
                physical_step,
                _max_abs(actions),
                getattr(self, "_rlinf_nonfinite_action_max_abs", None),
                getattr(self, "_rlinf_nonfinite_action_peak_step", None),
            )
            raise
        _update_env_peak(
            self,
            kind="state",
            value=obs.get("states"),
            physical_step=physical_step,
        )
        _assert_finite(rewards, stage=f"{state_stage}.rewards")
        _assert_finite(terminations, stage=f"{state_stage}.terminations")
        _assert_finite(truncations, stage=f"{state_stage}.truncations")
        return result

    IsaaclabBaseEnv.reset = checked_env_reset
    IsaaclabBaseEnv.step = checked_env_step
    logger.warning("Enabled W73 non-finite rollout/value diagnostics")


def _prepend_isaaclab_sources(source_root: Path) -> tuple[Path, ...]:
    """Put every revisioned Isaac Lab extension project first on sys.path."""

    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Isaac Lab source root does not exist: {source_root}")
    projects = tuple(
        path.resolve()
        for path in sorted(source_root.iterdir())
        if path.is_dir() and (path / "pyproject.toml").is_file()
    )
    if not projects:
        raise RuntimeError(
            f"Isaac Lab source root contains no extension projects: {source_root}"
        )
    project_strings = [str(path) for path in projects]
    sys.path[:] = project_strings + [
        path for path in sys.path if path not in project_strings
    ]
    return projects


def register() -> None:
    """Register IsaacLab extensions into RLinf.

    This function is called automatically by RLinf's Worker._load_user_extensions()
    when RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension is set.

    It performs the following registrations:
    1. Registers GR00T obs/action converters
    2. Registers GR00T data config
    3. Patches GR00T get_model for custom embodiment
    4. Registers task IDs from YAML config (env.*.init_params.id) into REGISTER_ISAACLAB_ENVS
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("isaaclab_contrib.rl.rlinf.extension: Registering IsaacLab extensions...")

    # Load config once and pass to all registration functions
    cfg = _get_isaaclab_cfg()

    _register_gr00t_converters(cfg)
    _patch_gr00t_get_model(cfg)
    _register_isaaclab_envs()
    if os.environ.get("RLINF_DEBUG_NONFINITE") == "true":
        _install_nonfinite_diagnostics()

    logger.info("isaaclab_contrib.rl.rlinf.extension: Registration complete.")


def _load_full_cfg() -> dict:
    """Load and cache the full YAML config from ``RLINF_CONFIG_FILE``.

    Raises:
        ValueError: If the ``RLINF_CONFIG_FILE`` environment variable is not set.

    Returns:
        The parsed YAML config as a nested dictionary.
    """
    global _full_cfg_cache
    if _full_cfg_cache is not None:
        return _full_cfg_cache
    config_file = os.environ.get("RLINF_CONFIG_FILE", "")
    if not config_file:
        raise ValueError("RLINF_CONFIG_FILE not set")
    with open(config_file) as f:
        _full_cfg_cache = yaml.safe_load(f)
    logger.info(f"Loaded full config from {config_file}")
    return _full_cfg_cache


def _get_isaaclab_cfg() -> dict:
    """Return the ``env.train.isaaclab`` section from the cached full config.

    Returns:
        The IsaacLab-specific configuration dictionary. Empty dict if the section is missing.
    """
    return _load_full_cfg().get("env", {}).get("train", {}).get("isaaclab", {})


def _patch_embodiment_tags(cfg: dict) -> None:
    """Add custom embodiment tag to RLinf's EmbodimentTag enum and mapping if needed.

    Reads ``embodiment_tag`` and ``embodiment_tag_id`` from the IsaacLab config section.
    Only adds the tag if it is not already present in RLinf's native registry.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    # GR00T uses embodiment tags to identify different robots.  Custom robots
    # (like G129+Dex3) need a unique tag string and numeric ID so that the
    # model's tokenizer can map them to the correct action/state dimensions.
    #
    # The numeric ID is the projector index in GR00T's Action Expert Module.
    # Known mapping (from gr00t/data/embodiment_tags.py):
    #   17 = oxe_droid, 24 = gr1, 26 = agibot_genie1, 31 = new_embodiment
    # Default 31 corresponds to the "new_embodiment" slot reserved for
    # fine-tuning on custom robots.
    embodiment_tag = cfg.get("embodiment_tag", "new_embodiment")
    tag_id = cfg.get("embodiment_tag_id", 31)

    # If tag is already in registry (native or previously added), skip
    if embodiment_tag in embodiment_tags.EMBODIMENT_TAG_MAPPING:
        logger.info(f"embodiment_tag '{embodiment_tag}' already registered")
        return
    # Add to enum
    tag_upper = embodiment_tag.upper().replace("-", "_")
    if not hasattr(embodiment_tags.EmbodimentTag, tag_upper):
        existing_members = {e.name: e.value for e in embodiment_tags.EmbodimentTag}
        existing_members[tag_upper] = embodiment_tag
        NewEmbodimentTag = Enum("EmbodimentTag", existing_members)

        embodiment_tags.EmbodimentTag = NewEmbodimentTag
        logger.info(f"Added EmbodimentTag.{tag_upper} = '{embodiment_tag}'")

    # Add to mapping
    embodiment_tags.EMBODIMENT_TAG_MAPPING[embodiment_tag] = tag_id
    logger.info(f"Added EMBODIMENT_TAG_MAPPING['{embodiment_tag}'] = {tag_id}")


def _load_n1d7_trocar_model(model_cfg, torch_dtype: torch.dtype) -> object:
    """Load GR00T N1.7 with the Trocar modality and normalization contract."""
    from gr00t.data.types import ModalityConfig
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
        redirect_qwen3_backbone_to_local,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    model_path = Path(model_cfg.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    backbone_path_value = OmegaConf.select(
        model_cfg, "backbone_model_path", default=None
    )
    if not backbone_path_value:
        raise ValueError("GR00T N1.7 Trocar requires backbone_model_path")
    backbone_path = Path(backbone_path_value).expanduser().resolve()
    if not backbone_path.is_dir():
        raise FileNotFoundError(f"Backbone model path does not exist: {backbone_path}")

    metadata_value = os.environ.get("W77_TROCAR_METADATA", "")
    if not metadata_value:
        raise RuntimeError("W77_TROCAR_METADATA is required for GR00T N1.7 Trocar")
    metadata_path = Path(metadata_value).expanduser().resolve()
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    try:
        statistics = {
            "new_embodiment": metadata["new_embodiment"]["statistics"]
        }
    except KeyError as error:
        raise ValueError(
            "Trocar metadata must contain new_embodiment.statistics"
        ) from error

    modality_configs = {
        "new_embodiment": {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=["left_wrist_view", "right_wrist_view", "room_view"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
            ),
            "action": ModalityConfig(
                delta_indices=list(range(16)),
                modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.human.action.task_description"],
            ),
        }
    }

    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    config = Gr00tN1d7Config.from_pretrained(str(model_path))
    config.action_dim = int(model_cfg.action_dim)
    config.tune_llm = False
    config.tune_visual = False
    config.tune_top_llm_layers = 0
    loading_kwargs = {"trust_remote_code": True, "local_files_only": True}
    with redirect_qwen3_backbone_to_local(config.model_name, str(backbone_path)):
        processor = Gr00tN1d7Processor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=False,
            image_crop_size=list(config.image_crop_size),
            image_target_size=list(config.image_target_size),
            shortest_image_edge=config.shortest_image_edge,
            crop_fraction=config.crop_fraction,
            random_rotation_angle=0,
            color_jitter_params=None,
            formalize_language=True,
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            max_action_horizon=config.action_horizon,
            apply_sincos_state_encoding=False,
            use_albumentations=False,
            use_relative_action=False,
            # N1.7 reserves projector 10 for custom post-training embodiments.
            embodiment_id_mapping={"new_embodiment": 10},
            transformers_loading_kwargs=loading_kwargs,
            exclude_state=False,
            state_dropout_prob=0.0,
            use_mean_std=False,
            letter_box_transform=False,
        )

    model = GR00T_N1_7_ForRLActionPrediction.from_pretrained(
        config=config,
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        backbone_model_path=str(backbone_path),
        torch_dtype=torch_dtype,
        embodiment_tag="new_embodiment",
        modality_config=modality_configs,
        modality_transform=processor,
        denoising_steps=model_cfg.denoising_steps,
        output_action_chunks=model_cfg.num_action_chunks,
        obs_converter_type=model_cfg.obs_converter_type,
        rl_head_config=model_cfg.rl_head_config,
    )

    # PPO trains the action/value path only. The raw backbone output is the
    # future feature-reuse boundary, so fail closed unless it is fully frozen.
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.to(torch_dtype)
    if model_cfg.rl_head_config.add_value_head:
        seed = model_cfg.get("value_head_init_seed", None)
        if seed is None:
            model.action_head.value_head._init_weights()
        else:
            from rlinf.utils.convergence_seed import (
                critic_digest,
                maybe_seeded_value_head_init,
            )

            maybe_seeded_value_head_init(model.action_head.value_head, seed)
            logger.info(
                "RLINF_CRITIC_INIT seed=%d digest=%s",
                int(seed),
                critic_digest(model.action_head.value_head),
            )
    if model_cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    trainable_backbone = sum(
        parameter.numel() for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )
    if trainable_backbone != 0:
        raise RuntimeError(
            f"GR00T N1.7 backbone must be frozen, found {trainable_backbone} trainable parameters"
        )
    logger.info(
        "Loaded GR00T N1.7 Trocar model model=%s backbone=%s metadata=%s",
        model_path,
        backbone_path,
        metadata_path,
    )
    return model


def _patch_gr00t_get_model(cfg: dict) -> None:
    """Monkeypatch RLinf's GR00T ``get_model`` to support custom ``data_config``.

    The patch is applied only when the user specifies a ``data_config_class`` in the
    YAML config. Embodiment tags are always ensured to be registered.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    # Always ensure embodiment tag is registered
    _patch_embodiment_tags(cfg)
    # Only patch get_model if user wants custom data_config
    data_config_class = cfg.get("data_config_class", "")
    if not data_config_class:
        logger.info("No data_config_class specified, using RLinf's default get_model")
        return

    import rlinf.models.embodiment.gr00t as rlinf_gr00t_mod

    def patched_get_model(model_cfg, torch_dtype=None) -> object:
        """Load a GR00T model with custom ``data_config`` and embodiment tag.

        Args:
            model_cfg: RLinf model configuration object containing ``model_path``,
                ``embodiment_tag``, ``denoising_steps``, ``num_action_chunks``,
                ``obs_converter_type``, and ``rl_head_config``.
            torch_dtype: The torch dtype for the model. Defaults to ``torch.bfloat16``.

        Raises:
            FileNotFoundError: If ``model_cfg.model_path`` does not exist.

        Returns:
            The loaded GR00T model instance.
        """
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        if str(model_cfg.model_type) == "gr00t_n1d7":
            return _load_n1d7_trocar_model(model_cfg, torch_dtype)

        # Handle custom embodiment (we only get here if tag was not natively supported)
        from gr00t.experiment.data_config import load_data_config
        try:
            from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
                GR00T_N1_5_ForRLActionPrediction,
            )
        except ModuleNotFoundError:
            from rlinf.models.embodiment.gr00t.gr00t_action_model import (
                GR00T_N1_5_ForRLActionPrediction,
            )
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity
        from rlinf.utils.patcher import Patcher

        # Apply RLinf's standard EmbodimentTag patches
        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()

        data_config = load_data_config(data_config_class)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        model_path = Path(model_cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # rl_model_path: optional path to an RLinf checkpoint with full_weights.pt
        rl_model_path = getattr(model_cfg, "rl_model_path", None)

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=model_cfg.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=model_cfg.denoising_steps,
            output_action_chunks=model_cfg.num_action_chunks,
            obs_converter_type=model_cfg.obs_converter_type,
            tune_visual=False,
            tune_llm=False,
            rl_head_config=model_cfg.rl_head_config,
        )

        if rl_model_path:
            rl_weights = Path(rl_model_path) / "actor" / "model_state_dict" / "full_weights.pt"
            if not rl_weights.exists():
                raise FileNotFoundError(
                    f"rl_model_path={rl_model_path}: cannot find full_weights.pt "
                    f"(tried directly and under actor/model_state_dict/)"
                )
            logger.info(f"Loading RL finetuned weights from {rl_weights}")
            state_dict = torch.load(rl_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)

        model.to(torch_dtype)
        if model_cfg.rl_head_config.add_value_head:
            # Opt-in deterministic critic-head seeding. This adapter's
            # patched_get_model REPLACES rlinf.get_model whenever a custom
            # data_config_class is set (as here), so the seeding MUST live here;
            # the worktree n1d5 patch is bypassed for this task. Default
            # (value_head_init_seed absent) -> exactly the prior unseeded
            # behavior, so all other experiments are unaffected.
            _vh_seed = model_cfg.get("value_head_init_seed", None)
            if _vh_seed is None:
                model.action_head.value_head._init_weights()
            else:
                from rlinf.utils.convergence_seed import (
                    critic_digest,
                    maybe_seeded_value_head_init,
                )

                maybe_seeded_value_head_init(
                    model.action_head.value_head, _vh_seed
                )
                logger.info(
                    f"RLINF_CRITIC_INIT seed={int(_vh_seed)} "
                    f"digest={critic_digest(model.action_head.value_head)}"
                )
        if model_cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)

        logger.info(f"Loaded GR00T model with embodiment_tag='{model_cfg.embodiment_tag}'")
        return model

    rlinf_gr00t_mod.get_model = patched_get_model
    logger.info(f"Patched get_model for data_config_class='{data_config_class}'")


def _register_gr00t_converters(cfg: dict) -> None:
    """Register GR00T obs/action converters for IsaacLab tasks.

    Reads ``obs_converter_type`` from the YAML config (``env.train.isaaclab.obs_converter_type``)
    and registers the corresponding observation and action conversion functions into
    RLinf's ``simulation_io`` registry.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    from rlinf.models.embodiment.gr00t import simulation_io

    obs_converter_type = cfg.get("obs_converter_type", "dex3")

    if obs_converter_type not in simulation_io.OBS_CONVERSION:
        simulation_io.OBS_CONVERSION[obs_converter_type] = _convert_isaaclab_obs_to_gr00t
        logger.info(f"Registered obs converter: {obs_converter_type}")

    action_registries = []
    for name in ("ACTION_CONVERSION_N1D5", "ACTION_CONVERSION_N1D7"):
        registry = getattr(simulation_io, name, None)
        if registry is not None:
            action_registries.append((name, registry))
    if not action_registries:
        legacy = getattr(simulation_io, "ACTION_CONVERSION", None)
        if legacy is None:
            raise RuntimeError("RLinf exposes no GR00T action conversion registry")
        action_registries.append(("ACTION_CONVERSION", legacy))
    for name, registry in action_registries:
        if obs_converter_type not in registry:
            registry[obs_converter_type] = _convert_gr00t_to_isaaclab_action
            logger.info("Registered %s action converter: %s", name, obs_converter_type)


def _convert_isaaclab_obs_to_gr00t(env_obs: dict) -> dict:
    """Convert IsaacLab env observations to GR00T format.

    Uses ``gr00t_mapping`` from the YAML config (``env.train.isaaclab.gr00t_mapping``)
    to map IsaacLab observation keys to GR00T-expected keys.

    Args:
        env_obs: Observation dictionary from ``_wrap_obs`` with the following keys:

            - ``"main_images"``: ``(B, H, W, C)`` torch tensor.
            - ``"extra_view_images"``: ``(B, N, H, W, C)`` torch tensor.
            - ``"states"``: ``(B, D)`` torch tensor.
            - ``"task_descriptions"``: list of strings.

    Returns:
        A dictionary with GR00T-formatted observations (numpy arrays with a time
        dimension, e.g. ``(B, T=1, H, W, C)``).
    """
    groot_obs = {}
    # Load mapping config from YAML or env var
    cfg = _get_isaaclab_cfg()
    gr00t_mapping = cfg.get("gr00t_mapping", {})
    video_mapping = gr00t_mapping.get("video", {})
    state_mapping = gr00t_mapping.get("state", [])
    # Convert main_images -> video.xxx
    if "main_images" in env_obs:
        main = env_obs["main_images"]
        gr00t_key = video_mapping.get("main_images", "video.room_view")
        if isinstance(main, torch.Tensor):
            # (B, H, W, C) -> (B, T=1, H, W, C)
            groot_obs[gr00t_key] = main.unsqueeze(1).cpu().numpy()
    # Convert extra_view_images -> video.xxx
    if "extra_view_images" in env_obs:
        extra = env_obs["extra_view_images"]  # (B, N, H, W, C)
        extra_keys = video_mapping.get("extra_view_images", [])
        if isinstance(extra, torch.Tensor):
            for i, key in enumerate(extra_keys):
                if i < extra.shape[1]:
                    # (B, H, W, C) -> (B, T=1, H, W, C)
                    groot_obs[key] = extra[:, i].unsqueeze(1).cpu().numpy()
    # Convert states -> state.xxx with slicing
    if "states" in env_obs and state_mapping:
        states = env_obs["states"]  # (B, D)
        if isinstance(states, torch.Tensor):
            states_np = states.unsqueeze(1).cpu().numpy()  # (B, T=1, D)
            for spec in state_mapping:
                gr00t_key = spec.get("gr00t_key")
                slice_range = spec.get("slice", [0, states_np.shape[-1]])
                if gr00t_key:
                    groot_obs[gr00t_key] = states_np[:, :, slice_range[0] : slice_range[1]]

    # Pass through task descriptions
    groot_obs["annotation.human.action.task_description"] = env_obs.get("task_descriptions", [])

    return groot_obs


def _convert_gr00t_to_isaaclab_action(action_chunk: dict, chunk_size: int = 1) -> np.ndarray:
    """Convert GR00T action output to IsaacLab env action format.

    Uses ``action_mapping`` from the YAML config (``env.train.isaaclab.action_mapping``)
    to apply optional prefix/suffix zero-padding to the concatenated action vector.

    Args:
        action_chunk: Dictionary of action arrays from GR00T, each with shape
            ``(B, T, D_i)``.
        chunk_size: Number of time steps to keep from the action chunk. Defaults to 1.

    Returns:
        Concatenated and padded action array with shape ``(B, chunk_size, D)``.
    """

    # Load mapping config from YAML or env var
    cfg = _get_isaaclab_cfg()
    action_mapping = cfg.get("action_mapping", {})
    prefix_pad = action_mapping.get("prefix_pad", 0)
    suffix_pad = action_mapping.get("suffix_pad", 0)

    # Concatenate all action parts
    action_parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
    action_concat = np.concatenate(action_parts, axis=-1)

    # Apply padding
    if prefix_pad > 0 or suffix_pad > 0:
        action_concat = np.pad(
            action_concat,
            ((0, 0), (0, 0), (prefix_pad, suffix_pad)),
            mode="constant",
            constant_values=0,
        )
    return action_concat


def _register_isaaclab_envs() -> None:
    """Register IsaacLab tasks into RLinf's REGISTER_ISAACLAB_ENVS map.

    Task IDs are read from ``env.train.init_params.id`` and
    ``env.eval.init_params.id`` in the YAML config.
    """
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    # Collect unique task IDs from the YAML config (train + eval)
    full_cfg = _load_full_cfg()
    env_cfg = full_cfg.get("env", {})
    task_ids: list[str] = []
    for section in ("train", "eval"):
        tid = env_cfg.get(section, {}).get("init_params", {}).get("id", "")
        if tid and tid not in task_ids:
            task_ids.append(tid)

    if not task_ids:
        logger.warning("No task IDs found in YAML config (env.*.init_params.id)")
        return

    logger.info(f"Tasks to register: {task_ids}")

    for task_id in task_ids:
        if task_id in REGISTER_ISAACLAB_ENVS:
            logger.debug(f"Task '{task_id}' already registered, skipping")
            continue

        # Create a generic wrapper class for this task
        env_class = _create_generic_env_wrapper(task_id)
        REGISTER_ISAACLAB_ENVS[task_id] = env_class
        logger.info(f"Registered IsaacLab task '{task_id}' for RLinf")

    logger.debug(f"REGISTER_ISAACLAB_ENVS now contains: {list(REGISTER_ISAACLAB_ENVS.keys())}")


def _create_generic_env_wrapper(task_id: str) -> type:
    """Create a generic wrapper class for an IsaacLab task.

    The wrapper class loads the task configuration in the environment child
    process and configures observation mapping accordingly.

    W68 enters Isaac Lab's kitless simulation context in that child process;
    it never starts AppLauncher, Kit, Vulkan, or RTX.

    Args:
        task_id: The gymnasium task ID.

    Returns:
        A class that inherits from IsaaclabBaseEnv.
    """
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    _task_id = task_id

    class IsaacLabGenericEnv(IsaaclabBaseEnv):
        """Generic environment wrapper for IsaacLab tasks.

        Config is read from the YAML file via ``_get_isaaclab_cfg()``.
        """

        def __init__(self, cfg, num_envs: int, seed_offset: int, total_num_processes: int, worker_info):
            """Initialize the generic IsaacLab environment wrapper.

            Args:
                cfg: RLinf environment configuration object.
                num_envs: Number of parallel environments.
                seed_offset: Seed offset for reproducibility.
                total_num_processes: Total number of worker processes.
                worker_info: RLinf worker metadata.
            """
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _record_metrics(self, step_reward, terminations, infos):
            """Override to use terminations (task completion) for success_once."""

            episode_info = {}
            self.returns += step_reward
            self.success_once = self.success_once | terminations.bool()
            episode_info["success_once"] = self.success_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
            infos["episode"] = episode_info
            return infos

        def _make_env_function(self) -> collections.abc.Callable:
            """Create the environment factory function.

            This function runs in a child process (via ``SubProcIsaacLabEnv``).
            All Isaac Lab-dependent imports happen here, in the child process.

            Returns:
                A callable that returns the environment and a simulation-context owner.
            """

            def make_env_isaaclab() -> tuple:
                """Create the IsaacLab environment inside the child process.

                Returns:
                    A tuple of ``(env, owner)``. ``owner.close()`` exits the
                    kitless simulation context through RLinf's existing
                    ``sim_app.close()`` compatibility call.
                """
                import gymnasium as gym

                source_root = Path(os.environ["W68_ISAACLAB_SOURCE_ROOT"])
                overlay_root = Path(os.environ["W68_OVERLAY_ROOT"])

                _prepend_isaaclab_sources(source_root)
                sys.path.insert(0, str(overlay_root))

                from isaaclab_tasks.utils import launch_simulation, resolve_task_config
                from w68_newton_trocar import TASK_ID, register_task

                if self.isaaclab_env_id != TASK_ID:
                    raise RuntimeError(
                        f"W68 Newton adapter expected task {TASK_ID}, got {self.isaaclab_env_id}"
                    )
                register_task()
                original_argv = sys.argv
                try:
                    sys.argv = [sys.argv[0], "presets=newton_mjwarp,newton_renderer"]
                    isaac_env_cfg, _ = resolve_task_config(self.isaaclab_env_id, "")
                finally:
                    sys.argv = original_argv

                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs
                isaac_env_cfg.seed = self.seed
                isaac_env_cfg.sim.device = "cuda:0"

                if float(isaac_env_cfg.rewards.update_stage.weight) != 1.0:
                    raise RuntimeError(
                        "W68 requires update_stage.weight=1.0 before environment creation"
                    )

                context = launch_simulation(
                    isaac_env_cfg,
                    {"visualizer": None, "visualizer_explicit": True},
                )
                context.__enter__()

                class SimulationContextOwner:
                    def __init__(self, simulation_context):
                        self._simulation_context = simulation_context
                        self._closed = False

                    def close(self) -> None:
                        if not self._closed:
                            self._closed = True
                            self._simulation_context.__exit__(None, None, None)

                owner = SimulationContextOwner(context)
                try:
                    env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg).unwrapped
                except Exception:
                    owner.close()
                    raise

                return env, owner

            return make_env_isaaclab

        def _wrap_obs(self, obs: dict) -> dict:
            """Convert IsaacLab observations to the RLinf format.

            The output format matches i4h's convention:

            - ``"main_images"``: ``(B, H, W, C)`` — single main camera.
            - ``"extra_view_images"``: ``(B, N, H, W, C)`` — stacked extra cameras.
            - ``"states"``: ``(B, D)`` — concatenated state vector.
            - ``"task_descriptions"``: ``list[str]`` — task descriptions.
            Config is read from the YAML file via :func:`_get_isaaclab_cfg`.

            Args:
                obs: Raw observation dictionary from the IsaacLab environment.

            Returns:
                A dictionary with observations mapped to the RLinf convention.
            """
            # import torch

            policy_obs = obs.get("policy", obs)
            camera_obs = obs.get("camera_images", {})

            cfg = _get_isaaclab_cfg()
            # Get task description from config
            task_desc = cfg.get("task_description", "") or self.task_description
            rlinf_obs = {
                "task_descriptions": [task_desc] * self.num_envs,
            }

            if not cfg:
                logger.warning("IsaacLab config is empty, returning minimal observation")
                return rlinf_obs

            # main_images: single camera key -> (B, H, W, C)
            main_key = cfg.get("main_images")
            if main_key and main_key in camera_obs:
                rlinf_obs["main_images"] = camera_obs[main_key]

            # extra_view_images: camera key(s) -> stack to (B, N, H, W, C)
            extra_keys = cfg.get("extra_view_images")
            if extra_keys:
                if isinstance(extra_keys, str):
                    extra_keys = [extra_keys]
                extra_imgs = [camera_obs[k] for k in extra_keys if k in camera_obs]
                if extra_imgs:
                    rlinf_obs["extra_view_images"] = torch.stack(extra_imgs, dim=1)

            # states: list of state specs -> concatenate to (B, D)
            # Each spec: string "key" or dict {"key": "...", "slice": [start, end]}
            state_specs = cfg.get("states")
            if state_specs:
                state_parts = []
                for spec in state_specs:
                    if isinstance(spec, str):
                        state = policy_obs.get(spec)
                        if state is not None:
                            state_parts.append(state)
                    elif isinstance(spec, dict):
                        state = policy_obs.get(spec.get("key"))
                        if state is not None:
                            slice_range = spec.get("slice")
                            if slice_range:
                                state = state[:, slice_range[0] : slice_range[1]]
                            state_parts.append(state)
                if state_parts:
                    rlinf_obs["states"] = torch.cat(state_parts, dim=-1)

            return rlinf_obs

        def add_image(self, obs: dict) -> np.ndarray | None:
            """Get image for video logging.

            Args:
                obs: Raw observation dictionary from the IsaacLab environment.

            Returns:
                A numpy array of shape ``(H, W, C)`` for the first environment, or
                ``None`` if no camera image is available.
            """
            camera_obs = obs.get("camera_images", {})
            cfg = _get_isaaclab_cfg()
            # Try main_images key, fallback to first available camera
            main_key = cfg.get("main_images")
            if main_key and main_key in camera_obs:
                return camera_obs[main_key][0].cpu().numpy()
            for img in camera_obs.values():
                return img[0].cpu().numpy()
            return None

    return IsaacLabGenericEnv
