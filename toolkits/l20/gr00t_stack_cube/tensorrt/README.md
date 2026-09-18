# GR00T N1.5 Stack Cube TensorRT Backbone

These tools export and qualify the fixed-shape L20 rollout backbone used by
the GR00T N1.5 Stack Cube workload. The qualified W06 mode keeps the
production PyTorch/FlashAttention ViT and runs only the frozen 12-layer Qwen
segment through a persistent TensorRT engine. The action/value head remains
PyTorch and can be updated normally.

## Artifact Flow

1. Run `probe_n15_backbone.py` to capture the exact true-B8 fixture.
2. Run `export_n15_backbone.py` to export static BF16-I/O ViT and LLM ONNX.
3. Run `build_n15_backbone.py` on an SM89 L20 to create and inspect plans.
4. Run `qualify_n15_backbone.py --components llm_only` against the retained
   fixture and engine receipt.

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
```

`rollout.enable_offload` must be `false`. Engine replacement happens after
the rollout checkpoint is loaded. The qualified mode loads one persistent
LLM engine/context and keeps all inputs, outputs, and embedding/scatter glue
CUDA-resident on the current PyTorch stream.

`components: full` remains an experimental diagnostic mode. W06 found that
the all-BF16 full backbone was faster but failed the frozen feature gate, while
an FP32-compute ViT improved numerical agreement but was slower than eager.
Neither full-backbone artifact is approved for PPO use.

This qualification is systems-only. PPO ratio/KL, value authority, and
learning convergence must be established by the follow-up training gate.
