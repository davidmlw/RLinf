GR00T N1.7 TensorRT Hybrid
==========================

This backend runs the frozen GR00T N1.7 ViT and language model with persistent
TensorRT engines. The trainable action head and value head remain in PyTorch,
so PPO can update their parameters without rebuilding or refitting TensorRT.
The Rollout worker publishes the exact TensorRT backbone features through
RLinf's pinned CUDA IPC path, and the Actor consumes those features for current
log-probability, value, and loss computation.

Support Boundary
----------------

The supported candidate has this composition:

* TensorRT: frozen ``vit.engine`` and ``llm_bf16.engine``.
* PyTorch eager: VLLN, VL self-attention, state/action encoders, DiT, action
  decoder, and value head.
* RLinf transport: ``borrowed_ipc_pinned`` rollout-feature reuse.
* Weight synchronization: ``action_head`` only; TensorRT plans remain loaded.

The implementation is Python and uses the TensorRT Python runtime. It does not
depend on Praxis, Poiesis, Rust, or AOTI. PT2 executors and a refittable
TensorRT DiT are experimental: they improve standalone inference latency but
have not passed the same-revision PPO ratio/KL gate. Do not use them for PPO
training without a new qualification receipt.

.. warning::

   TensorRT plans are specific to the TensorRT/CUDA/plugin and GPU environment.
   An incompatible plan, binding, receipt, static batch, sequence profile, or
   compute capability causes startup to fail. There is no silent eager fallback.

Prerequisites
-------------

The current artifact contract is intentionally narrow:

* Isaac-GR00T commit ``51d4c89f72fda44cbf77285c6a8114b52676b8a1``.
* ``nvidia/GR00T-N1.7-3B`` model revision
  ``2fc962b973bccdd5d8ce4f67cc63b264d6886495``.
* ``nvidia/Cosmos-Reason2-2B`` revision
  ``9ce19a195e423419c349abfc86fd07178b230561``.
* TensorRT ``10.15.1.29`` on an H100 (SM90).
* A genuine static batch of 8, three 224x224 camera views per row, and LLM
  sequence length 208. Running eight B1 calls is not equivalent.
* Trocar metadata with the 28-dimensional state/action ordering expected by
  ``trocar_b8_model_view.py``.

The machine-readable support and artifact contract is
``toolkits/eos/gr00t_trocar/tensorrt/hybrid-engine-support.json``.

Build the Engine Bundle
-----------------------

Build artifacts outside the Git checkout. The commands below use symbolic
paths so the same procedure works on a workstation or a scheduled GPU node.
The Python environment must contain the pinned GR00T export stack and
TensorRT builder.

.. code-block:: bash

   export RLINF_ROOT="$PWD"
   export GROOT_SOURCE=/path/to/Isaac-GR00T
   export GROOT_MODEL=/path/to/GR00T-N1.7-3B
   export GROOT_BACKBONE=/path/to/Cosmos-Reason2-2B
   export TROCAR_METADATA=/path/to/trocar/metadata.json
   export BUILD_ROOT=/path/to/gr00t-trt-b8-build
   export PYTHONPATH="$GROOT_SOURCE/scripts/deployment:$PYTHONPATH"

   python toolkits/eos/gr00t_trocar/tensorrt/trocar_b8_model_view.py \
     --model "$GROOT_MODEL" \
     --backbone "$GROOT_BACKBONE" \
     --metadata "$TROCAR_METADATA" \
     --output "$BUILD_ROOT/model-view"

   python toolkits/eos/gr00t_trocar/tensorrt/trocar_b8_fixture.py \
     --source "$GROOT_SOURCE" \
     --model "$BUILD_ROOT/model-view" \
     --metadata "$TROCAR_METADATA" \
     --output "$BUILD_ROOT/fixture"

   python toolkits/eos/gr00t_trocar/tensorrt/export_true_b8.py \
     --source "$GROOT_SOURCE" \
     --model "$BUILD_ROOT/model-view" \
     --collated "$BUILD_ROOT/fixture/collated-inputs.pt" \
     --fixture-receipt "$BUILD_ROOT/fixture/fixture.json" \
     --output "$BUILD_ROOT/onnx"

   python toolkits/eos/gr00t_trocar/tensorrt/build_true_b8.py \
     --onnx "$BUILD_ROOT/onnx" \
     --output "$BUILD_ROOT/engines" \
     --workspace 8192

The final engine directory must contain exactly ``vit.engine``,
``llm_bf16.engine``, ``export_metadata.json``, and
``rlinf-engine-receipt.json``. The receipt contains engine hashes, binding
tables, static-B8 assertions, the L=208 profile, and the no-fallback contract.
Keep the ONNX graphs and any external-data files with the build receipt even
though RLinf does not load them during training.

Qualify the Standalone Runtime
------------------------------

Run the standalone gate before enabling RLinf. It checks the complete
fixture-to-engine hash chain, persistent engine/context/buffer lifecycle,
CUDA-resident handoff, fixed-noise repeatability, finite outputs, and eager
versus hybrid normalized-action parity.

.. code-block:: bash

   python toolkits/eos/gr00t_trocar/tensorrt/standalone_true_b8.py \
     --source "$GROOT_SOURCE" \
     --model "$BUILD_ROOT/model-view" \
     --engines "$BUILD_ROOT/engines" \
     --collated "$BUILD_ROOT/fixture/collated-inputs.pt" \
     --raw "$BUILD_ROOT/fixture/raw-observation.npz" \
     --fixture-receipt "$BUILD_ROOT/fixture/fixture.json" \
     --export-receipt "$BUILD_ROOT/onnx/rlinf-export-receipt.json" \
     --engine-receipt "$BUILD_ROOT/engines/rlinf-engine-receipt.json" \
     --output "$BUILD_ROOT/standalone-qualification.json"

Do not continue unless the output receipt has ``"status": "passed"``. The
standalone gate verifies deployment actions only; it does not establish PPO
log-probability or value equivalence.

Enable the RLinf Candidate
--------------------------

Make the TensorRT Python packages visible through a dedicated overlay, then
bind the qualified engine directory and receipt hash:

.. code-block:: bash

   export RLINF_GROOT_TRT_RUNTIME_OVERLAY=/path/to/tensorrt-python-overlay
   export RLINF_GROOT_TRT_ENGINE_DIR="$BUILD_ROOT/engines"
   export RLINF_GROOT_TRT_ENGINE_RECEIPT_SHA256="$(
     sha256sum "$BUILD_ROOT/engines/rlinf-engine-receipt.json" | awk '{print $1}'
   )"
   export PYTHONPATH="$RLINF_GROOT_TRT_RUNTIME_OVERLAY:$PYTHONPATH"

Use
``toolkits/eos/gr00t_trocar/config-n1d7-hybrid-trt-eager-reuse-chunk16.yaml``
as the candidate configuration. Keep these settings unchanged for the first
qualification:

.. code-block:: yaml

   rollout:
     enable_torch_compile: false
     pinned_feature_verify_trajectory: false
     model:
       tensorrt_backbone:
         enabled: true
   actor:
     pre_update_same_revision_gate:
       enabled: true
     model:
       rollout_backbone_feature_transport: borrowed_ipc_pinned
   weight_syncer:
     state_dict_prefixes: [action_head]

For the EOS launcher, include exactly one ``rlinf-engine-receipt.json`` and one
``rlinf-tensorrt-overlay.json`` in ``provenance.files``. Put the overlay parent
in ``runtime.python_deps``. ``toolkits/eos/start_rlinf.py`` derives the three
``RLINF_GROOT_TRT_*`` variables from those immutable entries and passes them to
Ray workers.

Qualification Gates
-------------------

A retained RLinf trial must prove all of the following:

* ``RLINF_HYBRID_RUNTIME`` reports one engine load and one context per engine;
  revision adoption does not reload, rebuild, or refit either plan.
* Every Rollout feature block has the expected producer, owner, sample order,
  model revision, byte count, lease, and ACK; fallback count is zero.
* The Actor pre-update gate consumes the same pinned feature bytes as training.
  At the same policy revision, ratio and KL remain within the frozen thresholds
  in ``contract-n1d7-rlinf-hybrid-integration.json``.
* Losses and gradients are finite, and shutdown telemetry records both
  ``closing`` and ``closed`` states.

Use at least two outer steps for a functional receipt. For performance, use at
least five resident steps, treat step 0 as warmup, and summarize steps 1-4.
Report Rollout/Predict, concurrent Rollout+Env wall, and outer-step wall as
separate boundaries; never add overlapping worker timers.

Troubleshooting
---------------

``TensorRT overlay is absent from the configured Python dependencies``
   Add the overlay directory to the launcher's ``runtime.python_deps``. The
   manifest must be named ``rlinf-tensorrt-overlay.json``.

``engine receipt SHA-256 mismatch``
   Recompute ``RLINF_GROOT_TRT_ENGINE_RECEIPT_SHA256`` from the selected bundle.
   Do not edit a qualified receipt or mix files from different builds.

``true static B8`` or binding/profile failure
   Rebuild from the distinct-row B8 fixture. Do not pass an already-expanded B8
   visual capture through an exporter that expands batch a second time.

PPO ratio/KL failure with acceptable final-action parity
   Final actions are not a PPO correctness proof. Confirm exact feature reuse,
   policy revision, explicit noise, sample order, masks, and Actor/Rollout head
   equality. Keep the backend experimental until the PPO gate passes.
