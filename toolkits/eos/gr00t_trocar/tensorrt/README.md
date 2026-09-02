# GR00T N1.7 TensorRT Backbone

This directory owns the RLInf-side qualification contract for the GR00T N1.7
TensorRT frozen-backbone hybrid. The online implementation remains Python and
uses TensorRT's Python runtime with PyTorch CUDA tensors. Praxis and Poiesis are
design references only and are not runtime, build, or artifact dependencies.

The implementation is qualified in two distinct phases:

1. Reproduce the official B1 LIBERO seven-engine pipeline at Isaac-GR00T commit
   `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
2. Build and integrate a true-static-B8 Trocar backbone containing only the ViT
   and LLM TensorRT engines. The VLLN, action head, and value head stay in
   PyTorch and remain hot-updateable.

The official oracle is not the RLInf performance baseline. The matched RLInf
comparison is:

- control: PyTorch frozen backbone plus eager PyTorch action/value head;
- candidate: TensorRT frozen backbone plus the same eager PyTorch head.

Both arms enable the existing unused-logits optimization and disable rollout
feature reuse, making the backbone executor the only behavioral variable.

Large ONNX and TensorRT artifacts are not stored in Git. Unqualified build
outputs remain under the attempt directory. A bundle is promoted to the EOS
artifact cache only after manifest, deserialize, binding, numerical, and stream
interop gates pass.

The normative machine-readable contract is
`contract-n1d7-trocar-b8.json`. Any contract change must be reviewed before an
EOS qualification result can be retained.

## Official B1 Oracle

Run these commands on the EOS login node from a clean RLInf checkout. The first
command freezes the current Git revision and validates the image and inputs;
the second submits one exclusive H100 node. Add `--dry-run` to `submit` to
inspect the exact `sbatch` command without requesting resources.

The official lockfile selects TorchCodec 0.8.0, whose Linux wheel hard-loads
NVDEC even though this fixture is decoded on CPU. EOS compute containers do not
expose that optional video-driver library. The builder applies the compatible
TorchCodec 0.8.1 bugfix wheel by immutable path and SHA-256; all model, Torch,
Transformers, flash-attn, TensorRT, and export settings remain unchanged.

```bash
python toolkits/eos/gr00t_trocar/tensorrt/start_official_b1.py materialize \
  --template toolkits/eos/gr00t_trocar/tensorrt/site.official-b1.eos.template.json \
  --output /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W78/official-b1-site.json

python toolkits/eos/gr00t_trocar/tensorrt/start_official_b1.py submit \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W78/official-b1-site.json
```

The allocation creates `W78-official-b1-<job-id>/` under the attempt root. A
passing attempt contains the official logs, all seven ONNX graphs and engines,
and `qualification.json` with package, model, numerical, binding, and artifact
hash evidence. Failed and unqualified output stays attempt-local and is never
promoted to the artifact cache.
