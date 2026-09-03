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

"""Materialize the exact W77 Trocar processor as a local N1.7 model view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import model_view

EMBODIMENT = "new_embodiment"
EMBODIMENT_PROJECTOR_INDEX = 10
CAMERA_ORDER = ("left_wrist_view", "right_wrist_view", "room_view")
STATE_ACTION_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")
LANGUAGE_KEY = "annotation.human.action.task_description"
ACTION_DELTA_INDICES = tuple(range(16))


def _modality_config() -> dict[str, Any]:
    return {
        "video": {
            "delta_indices": [0],
            "modality_keys": list(CAMERA_ORDER),
        },
        "state": {
            "delta_indices": [0],
            "modality_keys": list(STATE_ACTION_ORDER),
        },
        "action": {
            "delta_indices": list(ACTION_DELTA_INDICES),
            "modality_keys": list(STATE_ACTION_ORDER),
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": [LANGUAGE_KEY],
        },
    }


def _trocar_statistics(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        statistics = metadata[EMBODIMENT]["statistics"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Trocar metadata must contain new_embodiment.statistics"
        ) from error
    if not isinstance(statistics, dict):
        raise ValueError("new_embodiment.statistics must be a JSON object")
    for modality in ("state", "action"):
        actual = tuple(statistics.get(modality, {}))
        if actual != STATE_ACTION_ORDER:
            raise ValueError(
                f"Trocar {modality} statistics order {actual} does not match "
                f"the W77 order {STATE_ACTION_ORDER}"
            )
    return statistics


def materialize_trocar_model_view(
    model_root: Path,
    backbone_root: Path,
    metadata_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create a symlink-backed model view with the W77 processor contract."""
    metadata_path = metadata_path.resolve(strict=True)
    metadata = model_view._read_object(metadata_path)
    statistics = _trocar_statistics(metadata)
    receipt = model_view.materialize_local_model_view(
        model_root, backbone_root, output_root
    )

    config_path = output_root / "config.json"
    processor_path = output_root / "processor_config.json"
    config = model_view._read_object(config_path)
    processor = model_view._read_object(processor_path)
    processor_kwargs = processor.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        raise ValueError("GR00T processor config omits processor_kwargs")
    modality_configs = processor_kwargs.get("modality_configs")
    all_statistics = processor_kwargs.get("statistics")
    if not isinstance(modality_configs, dict):
        modality_configs = {}
    if not isinstance(all_statistics, dict):
        all_statistics = {}
    modality_configs[EMBODIMENT] = _modality_config()
    all_statistics[EMBODIMENT] = statistics

    config.update(
        {
            "action_dim": 28,
            "tune_llm": False,
            "tune_visual": False,
            "tune_top_llm_layers": 0,
        }
    )
    processor_kwargs.update(
        {
            # AutoModel initialization still consults base embodiment entries.
            # Add Trocar without removing the checkpoint's existing modalities.
            "modality_configs": modality_configs,
            "statistics": all_statistics,
            "use_percentiles": False,
            "image_crop_size": list(config["image_crop_size"]),
            "image_target_size": list(config["image_target_size"]),
            "shortest_image_edge": config["shortest_image_edge"],
            "crop_fraction": config["crop_fraction"],
            "random_rotation_angle": 0,
            "color_jitter_params": None,
            "formalize_language": True,
            "model_type": "qwen",
            "max_state_dim": 132,
            "max_action_dim": 132,
            "max_action_horizon": 40,
            "apply_sincos_state_encoding": False,
            "use_albumentations": False,
            "use_relative_action": False,
            "embodiment_id_mapping": {EMBODIMENT: EMBODIMENT_PROJECTOR_INDEX},
            "exclude_state": False,
            "state_dropout_prob": 0.0,
            "use_mean_std": False,
            "letter_box_transform": False,
        }
    )
    model_view._write_object(config_path, config)
    model_view._write_object(processor_path, processor)

    receipt.update(
        {
            "schema": "rlinf.gr00t-n1d7-trocar-model-view.v1",
            "metadata": {
                "path": str(metadata_path),
                "sha256": model_view._sha256(metadata_path),
            },
            "processor_contract": {
                "embodiment": EMBODIMENT,
                "embodiment_projector_index": EMBODIMENT_PROJECTOR_INDEX,
                "camera_order": list(CAMERA_ORDER),
                "state_action_order": list(STATE_ACTION_ORDER),
                "language_key": LANGUAGE_KEY,
                "action_delta_indices": list(ACTION_DELTA_INDICES),
                "public_state_action_dim": 28,
                "model_state_action_width": 132,
                "generated_action_horizon": 40,
                "executed_action_horizon": 16,
            },
            "generated_hashes": {
                name: model_view._sha256(output_root / name)
                for name in sorted(model_view.GENERATED_FILES)
            },
        }
    )
    model_view._write_object(output_root / "rlinf-model-view.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = materialize_trocar_model_view(
            args.model, args.backbone, args.metadata, args.output
        )
    except Exception as error:
        print(f"W80 model-view creation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
