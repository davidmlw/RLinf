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
top-level paths. Torch, TensorRT, NumPy and pandas must remain image/runtime
authorities and may not be shadowed by that overlay.

## W96 L20 launch gate

`l20_vulkan_launcher.py` replaces the historical W88 launcher for W96. A site
JSON using schema `rlinf.w96.l20-vulkan-site/v1` binds:

- the pinned image reference and canonical immutable image receipt;
- the exact runtime spec and nine-manifest set;
- immutable RLinf and Isaac-GR00T sources, Python and TensorRT overlays,
  config/override roots, asset seed and the combined model input bundle;
- the Docker client hash, durable run root and Lustre quota mount; and
- a direct Git-tree attestation for the RLinf source bundle.

The combined model manifest must cover both `GR00T-N1.7-3B` and its local
`Cosmos-Reason2-2B` backbone. A model found only in `/tmp`, a path outside the
verified bundle, or an online Hugging Face fallback is rejected.

Build and verify the source attestation from a real Git checkout. Verification
reads the materialized files directly and never invokes `git -C` inside the
bundle, which intentionally has no `.git` directory:

```bash
python toolkits/gr00t_trocar/w95/git_tree_attestation.py build \
  --repo /path/to/RLinf \
  --revision <exact-commit> \
  --output /durable/receipts/rlinf-git-tree.json
python toolkits/gr00t_trocar/w95/git_tree_attestation.py verify \
  --root /durable/immutable/rlinf-source \
  --attestation /durable/receipts/rlinf-git-tree.json
```

The launch sequence is deliberately staged:

```bash
python toolkits/gr00t_trocar/w95/l20_vulkan_launcher.py validate \
  --site /durable/W96/l20-site.json
python toolkits/gr00t_trocar/w95/l20_vulkan_launcher.py plan \
  --site /durable/W96/l20-site.json \
  --phase q1 \
  --run-root /durable/runs/W96-q1
python toolkits/gr00t_trocar/w95/l20_vulkan_launcher.py q1 \
  --site /durable/W96/l20-site.json \
  --run-root /durable/runs/W96-q1
python toolkits/gr00t_trocar/w95/l20_vulkan_launcher.py q2 \
  --site /durable/W96/l20-site.json \
  --q1-receipt /durable/runs/W96-q1/q1-receipt.json \
  --run-root /durable/runs/W96-q2
```

Q1 rechecks at least 80 GiB of HOME headroom, eight idle L20 SM89 GPUs and the
canonical image before collecting driver, `libcuda`, NVIDIA Vulkan ICD/library
and import-origin receipts. It imports neither Isaac Lab nor Ray. Q2 repeats
the host preflight and the same in-container runtime-origin probe, consumes the
matching Q1 receipt, runs exactly one eager true-B8 chunk16 GR00T call, then
starts one local Assemble-Trocar Vulkan environment and executes one
zero-action turn. Q2 does not load a TensorRT engine, run Nsys or start Ray.
Both phases force the NVIDIA ICD through `VK_DRIVER_FILES`. Each attempt
receives a fresh writable asset cache copied from the immutable seed, while
source/model/config inputs remain read-only. Commands, configuration,
environment, container inspection, logs and output receipts remain under the
attempt directory. A no-GPU cleanup container normalizes all attempt output to
the invoking host UID/GID so failed probes do not leave root-owned artifacts.
