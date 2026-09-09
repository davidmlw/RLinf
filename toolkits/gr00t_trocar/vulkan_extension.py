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

"""Compose the original Vulkan Trocar environment with the N1.7 model loader.

The fixed Trocar runtime owns AppLauncher, PhysX, RTX cameras, task
registration, and the reward-stage fix. The N1.7 EOS adapter owns only the
model/processor contract used here. Importing its loader does not register the
Newton environment.
"""

import torch

from isaaclab_contrib.rl.rlinf import extension as _vulkan
from toolkits.eos.gr00t_trocar import w68_rlinf_extension as _n1d7


_base_register_converters = _vulkan._register_gr00t_converters


def _register_converters(cfg: dict) -> None:
    _base_register_converters(cfg)

    from rlinf.models.embodiment.gr00t import simulation_io

    obs_converter_type = cfg.get("obs_converter_type", "dex3")
    registry = getattr(simulation_io, "ACTION_CONVERSION_N1D7", None)
    if registry is None:
        raise RuntimeError("RLinf does not expose ACTION_CONVERSION_N1D7")
    if obs_converter_type not in registry:
        registry[obs_converter_type] = _vulkan._convert_gr00t_to_isaaclab_action


def _patch_n1d7_get_model(cfg: dict) -> None:
    _vulkan._patch_embodiment_tags(cfg)
    data_config_class = cfg.get("data_config_class", "")
    if data_config_class != "n1d7_trocar":
        raise RuntimeError(
            "The Vulkan N1.7 adapter requires data_config_class=n1d7_trocar"
        )

    import rlinf.models.embodiment.gr00t as rlinf_gr00t

    def get_model(model_cfg, torch_dtype=None):
        if str(model_cfg.model_type) != "gr00t_n1d7":
            raise RuntimeError(
                "The Vulkan N1.7 adapter only supports model_type=gr00t_n1d7"
            )
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        return _n1d7._load_n1d7_trocar_model(model_cfg, torch_dtype)

    rlinf_gr00t.get_model = get_model


def register() -> None:
    """Install N1.7 model hooks, then run the original Vulkan registration."""
    _vulkan._register_gr00t_converters = _register_converters
    _vulkan._patch_gr00t_get_model = _patch_n1d7_get_model
    _vulkan.register()
