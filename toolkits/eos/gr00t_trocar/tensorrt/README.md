# GR00T N1.7 Official TensorRT Oracle

This directory first qualifies the official GR00T N1.7 full-TensorRT B1 LIBERO
pipeline for W79. Praxis and Poiesis are read-only design references and are not
runtime, build, or artifact dependencies.

W79 only reproduces the official seven-engine pipeline at Isaac-GR00T commit
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`. The true-static-B8 Trocar hybrid,
RLInf integration, PPO correctness and performance A/B remain separate W78
work after this oracle passes.

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
The allocation also installs the Ubuntu 24.04 FFmpeg 6 runtime into its
ephemeral container and records exact `dpkg` versions in `ffmpeg.json`.

```bash
python toolkits/eos/gr00t_trocar/tensorrt/start_official_b1.py materialize \
  --template toolkits/eos/gr00t_trocar/tensorrt/site.official-b1.eos.template.json \
  --output /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W79/official-b1-site.json

python toolkits/eos/gr00t_trocar/tensorrt/start_official_b1.py submit \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W79/official-b1-site.json
```

The allocation creates `W79-official-b1-r1-<job-id>/` under the attempt root. A
passing attempt contains the builder/video probe, official logs, all seven ONNX
graphs and engines, and `qualification.json` with package, model, numerical,
binding and artifact-hash evidence. Failed output stays attempt-local and never
overwrites a qualified receipt.
