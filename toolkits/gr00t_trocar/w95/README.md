# GR00T N1.7 L20 experiment contract

This directory is the launch gate for the W95 experiment series. It binds the
RLinf `0f9ea98c` lineage to one GR00T N1.7, true-B8, chunk16, eight-L20,
PhysX/Vulkan workload.

Two Trainer execution profiles are intentionally distinct:

- `absolute_correctness_b8`: Actor microbatch 8 matches each Rollout B8 policy
  call. Only this profile may receive an absolute same-revision PPO disposition.
- `baseline_relative_throughput_b32`: Actor microbatch 32 is a performance
  profile. It may compare matched arms, but a B8 replay cannot qualify its
  optimizer path and it must not be labeled absolute `PPO PASS`.

Render and validate a config:

```bash
python toolkits/gr00t_trocar/w95/contract.py render \
  --base toolkits/gr00t_trocar/config-n1d7-vulkan-control.yaml \
  --profile absolute_correctness_b8 \
  --arm all_off \
  --output ./tmp/W96/config-b8-all-off.yaml

python toolkits/gr00t_trocar/w95/contract.py validate \
  --config ./tmp/W96/config-b8-all-off.yaml \
  --profile absolute_correctness_b8 \
  --arm all_off \
  --receipt ./tmp/W96/config-b8-all-off.receipt.json
```

The `manifest` command additionally requires a clean source worktree rooted in
the recorded lineage. Pass every external image/model/task/artifact receipt as
an `--evidence` argument. Missing evidence, a dirty source, a wrong ancestor or
an invalid config fails closed.

The performance policy is inherited from W71 v1: run five complete outer steps,
keep zero-based step 0 as warmup, and summarize steps 1-4 by mean, sample
standard deviation and CV. Overlapping Rollout and Env intervals are unioned or
intersected; they are never added.

Nsys attempts are separate from clean timing. Full-step captures use
CUDA/NVTX/OSRT for each role rank0. Vulkan and GPU metrics use a separate short
Env-only attempt. Profiled samples never enter headline performance means.

Immutable directory inputs use `tree_manifest.py`. The manifest records every
relative path, file type or symlink target, size, mode and SHA-256, and its
verification fails on additions, removals or metadata/content changes:

```bash
python toolkits/gr00t_trocar/w95/tree_manifest.py build \
  --root /path/to/immutable-tree \
  --output /path/to/manifests/tree.json
python toolkits/gr00t_trocar/w95/tree_manifest.py verify \
  --root /path/to/immutable-tree \
  --manifest /path/to/manifests/tree.json

python toolkits/gr00t_trocar/w95/tree_manifest.py materialize \
  --source /read-only/source/tree \
  --destination /owned/staging/tree \
  --manifest /path/to/manifests/tree.json
```

The manifest itself must live outside the tree so it cannot recursively hash
itself. Per-run writable scratch and output trees must never use an immutable
input manifest.

W95 L20 runs use
`runtime-spec-n1d7-l20-w88.json`: the pinned W88 image owns
`torch==2.10.0+cu128`, the Python overlay may not contain Torch or TensorRT,
and TensorRT 10.15.1.29 is mounted separately. The
`toolkits/eos/gr00t_trocar/runtime-spec-n1d7.json` Torch 2.11 environment is
scoped to EOS/H100/Newton and is not a W95 L20 runtime authority.
`python-overlay-requirements-w88.txt` reconstructs the final W88 overlay from
the retained install logs. Qualification first hashes every downloaded wheel,
then installs offline with `--no-deps` and rejects forbidden distributions or
top-level paths.
