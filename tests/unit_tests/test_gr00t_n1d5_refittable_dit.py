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

import inspect
from pathlib import Path

from rlinf.models.embodiment.gr00t.gr00t_n1d5 import tensorrt_dit
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    GR00T_N1_5_ForRLActionPrediction,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_n1d5_refittable_dit_freezes_the_production_b8_abi() -> None:
    assert tensorrt_dit._EXPECTED_INPUTS == {
        "sa_embs": ([8, 49, 1536], "BF16"),
        "vl_embs": ([8, 570, 2048], "BF16"),
        "timestep": ([8], "INT64"),
    }
    assert tensorrt_dit._EXPECTED_OUTPUT == ([8, 49, 1024], "BF16")
    assert tensorrt_dit._EXPECTED_REFIT_TENSORS == 232


def test_n1d5_runtime_rejects_n1d7_mask_inputs() -> None:
    source = inspect.getsource(tensorrt_dit.RefittableTensorRTDiT.__call__)

    assert '"image_mask"' not in source
    assert '"backbone_attention_mask"' not in source
    assert "encoder_attention_mask is not None" in source


def test_n1d5_model_exposes_online_refit_lifecycle() -> None:
    enable = inspect.getsource(GR00T_N1_5_ForRLActionPrediction.enable_tensorrt_dit)
    verify = inspect.getsource(
        GR00T_N1_5_ForRLActionPrediction.verify_online_update_contract
    )
    close = inspect.getsource(GR00T_N1_5_ForRLActionPrediction.close_hybrid_runtime)

    assert "online_refit" in enable
    assert "RefittableTensorRTDiT" in enable
    assert "verify_revision(revision)" in verify
    assert "tensorrt_dit.close()" in close


def test_rollout_loads_checkpoint_before_tensorrt_dit() -> None:
    source = inspect.getsource(MultiStepRolloutWorker.init_worker)

    assert source.index("self.hf_model.load_state_dict") < source.index(
        "enable_tensorrt_dit(tensorrt_dit_config)"
    )


def test_rollout_refits_before_publishing_revision() -> None:
    source = inspect.getsource(MultiStepRolloutWorker.sync_model_from_actor)

    assert source.index("verify_online_update(applied_version)") < source.index(
        "self.version = applied_version"
    )


def test_n1d5_export_materializes_inputs_after_inference_scope() -> None:
    source = Path(
        "toolkits/l20/gr00t_stack_cube/tensorrt/export_refittable_dit_b8.py"
    ).read_text(encoding="utf-8")

    inference_scope = source.index("with torch.inference_mode():")
    hook_cleanup = source.index("hook.remove()", inference_scope)
    materialization = source.index(
        "capture.inputs = _materialize_export_inputs(capture.inputs)"
    )
    export_call = source.index("torch.onnx.export(")
    assert hook_cleanup < materialization < export_call
    assert "torch.is_inference(value)" in source
