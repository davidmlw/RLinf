# GR00T N1.5 Stack Cube TensorRT Hybrid

These tools export and qualify the fixed-shape L20 rollout backbone used by
the GR00T N1.5 Stack Cube workload. The qualified W06 mode keeps the
production PyTorch/FlashAttention ViT and can run the frozen 12-layer Qwen
segment through a persistent TensorRT engine. The optional Action Head backend
replaces only the trainable DiT with a double-slot refittable TensorRT engine;
the encoders, decoder, value head, and parameter authority remain in PyTorch.

## Artifact Flow

1. Run `probe_n15_backbone.py` to capture the exact true-B8 fixture.
2. Run `export_n15_backbone.py` to export static BF16-I/O ViT and LLM ONNX.
3. Run `build_n15_backbone.py` on an SM89 L20 to create and inspect plans.
4. Run `qualify_n15_backbone.py --components llm_only` against the retained
   fixture and engine receipt.
5. Run `export_refittable_dit_b8.py` to export the static-B8 BF16 DiT.
6. Run `refittable_dit_contract.py` to bind every ONNX initializer to the 232
   authoritative `action_head.model.*` checkpoint tensors.
7. Run `build_refittable_dit_b8.py` on an SM89 L20 with TensorRT refit enabled.
8. Run `qualify_n15_refittable_dit.py` to compare full-TRT-Backbone/eager-DiT
   against full-TRT-Backbone/refittable-TRT-DiT and exercise revision adoption.

Every output directory must be new. The tools fail closed on fixture shape,
engine binding, TensorRT version, SM version, receipt hash, and numerical
qualification drift.

## RLInf Configuration

Add the following under `rollout.model`; use the SHA-256 from the retained
engine receipt rather than the example placeholder.

```yaml
tensorrt_backbone:
  enabled: true
  components: llm_only
  engine_dir: /absolute/path/to/qualified/engines
  receipt_path: /absolute/path/to/qualified/engines/rlinf-engine-receipt.json
  receipt_sha256: <sha256>
  static_batch_size: 8
  image_views: 2
  image_batch_size: 16
  sequence_length: 570
  runtime_version: 10.15.1.29
  runtime_distribution: tensorrt-cu12
  compute_capability: [8, 9]

tensorrt_dit:
  enabled: true
  online_refit: true
  engine_path: /absolute/path/to/dit_bf16_refit.engine
  receipt_path: /absolute/path/to/rlinf-refittable-dit-engine-receipt.json
  receipt_sha256: <sha256>
  parameter_map_path: /absolute/path/to/refittable-dit-parameter-map.json
  parameter_map_sha256: <sha256>
  source_digest_revision_0: <sha256>
  revision: 0
  runtime_version: 10.15.1.29
  runtime_distribution: tensorrt-cu12
  compute_capability: [8, 9]
  lineage_receipt_mode: gpu_transform_validation
  probe_each_revision: true
  minimum_probe_cosine: 0.999
  maximum_probe_relative_l2: 0.05
  minimum_free_device_bytes: 8589934592
  ppo_authority_status: unqualified_requires_convergence_validation
  shadow_eager: false
```

`rollout.enable_offload` must be `false`. Engine replacement happens after
the rollout checkpoint is loaded. The qualified mode loads one persistent
LLM engine/context and keeps all inputs, outputs, and embedding/scatter glue
CUDA-resident on the current PyTorch stream.

`components: full` remains an experimental mode. W06 found that the all-BF16
full backbone was faster but exceeded the original conservative feature gate.
It is retained for controlled convergence testing rather than being rejected
solely by that intermediate-feature threshold.

This qualification is systems-only. PPO ratio/KL, value authority, and
learning convergence must be established by the follow-up training gate.
