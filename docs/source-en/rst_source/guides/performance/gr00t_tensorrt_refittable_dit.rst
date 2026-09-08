Experimental GR00T Refittable TensorRT DiT
===========================================

This backend extends the supported GR00T TensorRT hybrid with an online
refittable TensorRT DiT. It keeps the frozen ViT and language-model TensorRT
engines, executes DiT in a double-buffered TensorRT runtime, and leaves the
remaining action and value heads in PyTorch.

.. warning::

   This backend is an approximate-behavior systems experiment. Deployment
   action parity and online refit lifecycle checks passed, but the
   same-revision PPO ratio/KL identity gate did not. Do not use it as a PPO
   training default or as convergence evidence. Use the eager PyTorch Action
   Head described in :doc:`gr00t_tensorrt_hybrid` for qualified PPO trials.

Runtime Design
--------------

The Rollout worker owns two persistent TensorRT DiT engine/context slots. A
new Actor revision is transformed into the engine's refit layout on GPU,
loaded into the inactive slot, checked on fixed live probes, and adopted only
at a revision boundary. The old slot remains active until the candidate is
qualified. The ViT and language-model plans are never rebuilt, reloaded, or
refitted during this operation.

The implementation uses Python, PyTorch CUDA tensors, and the TensorRT Python
runtime. It does not depend on Praxis, Poiesis, Rust, or AOTI.

Build and Qualify the DiT Bundle
--------------------------------

First build and qualify the frozen TensorRT Backbone by following
:doc:`gr00t_tensorrt_hybrid`. Reuse its true-static-B8 fixture and model view,
then export a refittable DiT graph:

.. code-block:: bash

   export RLINF_ROOT="$PWD"
   export GROOT_SOURCE=/path/to/Isaac-GR00T
   export BUILD_ROOT=/path/to/gr00t-trt-b8-build
   export DIT_ROOT=/path/to/refittable-dit-build
   export DIT_QUALIFICATION=/path/to/refit-qualification

   python toolkits/eos/gr00t_trocar/tensorrt/export_refittable_dit_b8.py \
     --source "$GROOT_SOURCE" \
     --model "$BUILD_ROOT/model-view" \
     --collated "$BUILD_ROOT/fixture/collated-inputs.pt" \
     --fixture-receipt "$BUILD_ROOT/fixture/fixture.json" \
     --output "$DIT_ROOT/onnx"

   python toolkits/eos/gr00t_trocar/tensorrt/refittable_dit_contract.py \
     --checkpoint "$BUILD_ROOT/model-view" \
     --model-config "$BUILD_ROOT/model-view/config.json" \
     --onnx "$DIT_ROOT/onnx/dit_bf16.onnx" \
     --output "$DIT_ROOT/refittable-dit-parameter-map.json"

   python toolkits/eos/gr00t_trocar/tensorrt/build_refittable_dit_b8.py \
     --onnx "$DIT_ROOT/onnx/dit_bf16.onnx" \
     --parameter-map "$DIT_ROOT/refittable-dit-parameter-map.json" \
     --output "$DIT_ROOT/engine" \
     --workspace-mib 8192

Run the device lifecycle qualification with the same model and fixture:

.. code-block:: bash

   python toolkits/eos/gr00t_trocar/tensorrt/refit_dit_lifecycle_probe.py \
     --source "$GROOT_SOURCE" \
     --model "$BUILD_ROOT/model-view" \
     --collated "$BUILD_ROOT/fixture/collated-inputs.pt" \
     --engine "$DIT_ROOT/engine/dit_bf16_refit.engine" \
     --engine-receipt \
       "$DIT_ROOT/engine/rlinf-refittable-dit-engine-receipt.json" \
     --parameter-map "$DIT_ROOT/refittable-dit-parameter-map.json" \
     --output "$DIT_QUALIFICATION"

Bind the build and lifecycle receipts before starting RLinf. This command
rejects missing artifacts, mixed builds, altered hashes, weakened numerical
thresholds, or a missing revision-zero source digest:

.. code-block:: bash

   python toolkits/eos/gr00t_trocar/tensorrt/verify_refittable_dit_bundle.py \
     --build-root "$DIT_ROOT" \
     --qualification "$DIT_QUALIFICATION/refit-lifecycle-receipt.json" \
     --output "$DIT_QUALIFICATION/runtime-bundle.json"

Run the Diagnostic Mode
-----------------------

The diagnostic mode executes TensorRT DiT at revision zero while retaining an
eager shadow for parity measurements. It does not perform online refits:

.. code-block:: bash

   export RLINF_GROOT_TRT_DIT_ROOT="$DIT_ROOT"
   export RLINF_GROOT_TRT_DIT_QUALIFICATION=\
"$DIT_QUALIFICATION/refit-lifecycle-receipt.json"
   export RLINF_GROOT_TRT_DIT_DIAGNOSTIC=1
   export RLINF_GROOT_TRT_DIT_ONLINE=0
   bash toolkits/eos/gr00t_trocar/run_n1d7_hybrid.sh

Run the Experimental Online Refit
---------------------------------

Online refit is opt-in and fail-closed. It cannot be enabled together with the
diagnostic mode:

.. code-block:: bash

   export RLINF_GROOT_TRT_DIT_DIAGNOSTIC=0
   export RLINF_GROOT_TRT_DIT_ONLINE=1
   export RLINF_GROOT_TRT_DIT_LINEAGE_MODE=qualification_sha256
   bash toolkits/eos/gr00t_trocar/run_n1d7_hybrid.sh

For the EOS launcher, include one
``rlinf-refittable-dit-engine-receipt.json``, one
``refittable-dit-parameter-map.json``, and one
``refit-lifecycle-receipt.json`` in ``provenance.files``. The launcher derives
``RLINF_GROOT_TRT_DIT_ROOT`` and
``RLINF_GROOT_TRT_DIT_QUALIFICATION``. Select diagnostic or online mode
explicitly; artifact presence alone never enables this backend.

Required Evidence
-----------------

Retain the bundle verification output, per-revision source and staging
digests, candidate probe metrics, slot transitions, refit latency, TensorRT
load/context counts, and shutdown state. Any mismatch must abort adoption;
there is no silent eager fallback.

The observed ratio/KL failure is a correctness boundary, not a tunable warning.
Promotion requires a new same-revision PPO receipt showing behavior/current
ratio approximately one and KL approximately zero across the complete
minibatch, followed by learning and convergence validation.
