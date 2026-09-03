# GR00T N1.7 TensorRT Qualification

This directory first qualifies the official GR00T N1.7 full-TensorRT B1 LIBERO
pipeline for W79. Praxis and Poiesis are read-only design references and are not
runtime, build, or artifact dependencies.

W79 reproduces the official seven-engine B1 LIBERO pipeline at Isaac-GR00T
commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`. W80 then qualifies the
true-static-B8 Trocar hybrid with TensorRT ViT/LLM and a resident PyTorch action
head. RLInf Rollout integration, revision adoption, PPO statistics and learning
correctness remain W78 work after these standalone gates pass.

Large ONNX and TensorRT artifacts are not stored in Git. Unqualified build
outputs remain under the attempt directory. A bundle is promoted to the EOS
artifact cache only after manifest, deserialize, binding, numerical, and stream
interop gates pass.

The reviewed W79 work item is the normative execution contract. The
machine-readable LFS input contract is `libero-b1-lfs.json`.

## Official B1 Oracle

Run these commands on the EOS login node from a clean RLInf checkout. The first
command freezes the current Git revision and validates the image and inputs;
the second submits one exclusive H100 node. Add `--dry-run` to `submit` to
inspect the exact `sbatch` command without requesting resources.

The official lockfile selects TorchCodec 0.8.0, whose Linux wheel hard-loads
`libnvcuvid`. The builder applies the Torch-2.9-compatible TorchCodec 0.8.1
bugfix wheel, then proves the exact interpreter/import/native-library path and
decodes a retained LIBERO MP4 through GR00T's CPU utility before ONNX export.
All model, Torch, Transformers, flash-attn, TensorRT, and export settings remain
unchanged.
The allocation also installs the Ubuntu 24.04 FFmpeg 6 runtime and Python 3.12
runtime/development libraries into its ephemeral container, then records exact
`dpkg` versions in `ffmpeg.json`.

```bash
python toolkits/eos/gr00t_trocar/tensorrt/start_official_b1.py materialize \
  --template toolkits/eos/gr00t_trocar/tensorrt/site.official-b1.eos.template.json \
  --output /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W79/official-b1-site.json

python toolkits/eos/gr00t_trocar/tensorrt/start_official_b1.py submit \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W79/official-b1-site.json
```

After the official attempt passes, record the independent component arrays and
resident public whole-call boundary without rebuilding the plans:

```bash
python toolkits/eos/gr00t_trocar/tensorrt/start_official_b1.py resident-submit \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W79/official-b1-resident-site.json \
  --oracle-attempt /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W79/W79-official-b1-r6-JOB_ID
```

The allocation creates `W79-official-b1-rN-<job-id>/` under the attempt root. A
passing attempt contains the builder/video probe, official logs, all seven ONNX
graphs and engines, and `qualification.json` with package, model, numerical,
binding and artifact-hash evidence. Failed output stays attempt-local and never
overwrites a qualified receipt.

## True-B8 Trocar Hybrid

The W80 path is split into reproducible stages:

- `trocar_b8_model_view.py` creates the immutable Trocar model view.
- `trocar_b8_fixture.py` captures one distinct-row, distinct-camera B8 input.
- `export_true_b8.py` exports ViT and LLM ONNX graphs from that exact fixture.
- `build_true_b8.py` builds and qualifies static-B8 `vit.engine` and
  `llm_bf16.engine`.
- `standalone_true_b8.py` verifies the provenance chain, numerical agreement,
  persistent engine lifecycle and resident performance.

The standalone runner records both the official CPU-collated entry point and a
common CUDA-resident boundary. The common boundary starts with preloaded
contiguous CUDA tensors and one explicit initial flow-noise tensor, and ends at
the normalized deployment action. It measures these direct, alternating AB/BA
pairs with CUDA events:

1. PyTorch eager versus TensorRT backbone plus eager head.
2. TensorRT backbone plus eager head versus the same backbone with compiled DiT.
3. PyTorch eager versus TensorRT backbone plus compiled DiT.

Raw preprocessing, H2D transfer, implicit RNG generation, public action decode,
and PPO transition noise/logprob/value are intentionally outside this boundary.
Compile time and graph creation are lifecycle metrics, not resident inference.
The runner fails if the explicit-noise path differs from the official deployment
path, advances RNG, recompiles during measurement, or executes a different
number of TensorRT calls than the phase contract requires.
