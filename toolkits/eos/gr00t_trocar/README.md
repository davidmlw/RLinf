# EOS GR00T Trocar Baseline

This directory owns the reproducible RLinf control represented by the
W71-before arm on EOS H100. It uses
RLinf base `0f9ea98c`, GR00T N1.5, the Newton/MJWarp Trocar task, and the
production action-chunk-16 PPO configuration. Rollout backbone feature reuse
and the other local performance features are absent.

## Runtime

RLinf source is never built into the runtime image or installed into the
shared Python environment. The EOS checkout at the exact site-manifest
revision is placed first on `PYTHONPATH` and executed directly.

The system image is the proven CUDA 12.8 compatibility substrate, republished
under the RLinf-owned GitLab reference
`liweim/image/rlinf-eos-system:cuda128-ubuntu2404-be095dc`. It supplies Ubuntu,
CUDA compatibility libraries, system dependencies and `uv`, but no experiment
Python environment. `prepare_runtime.sh` materializes the RLinf-owned shared
environment once at
`rlinf-workspace/envs/gr00t-newton-py312-cu128-v4/`. The committed runtime spec,
exact source revisions, package freeze and generated manifest identify it;
later tickets can reuse it without rebuilding an image or reinstalling Python
packages. `uv` also keeps its managed Python distribution, including matching
development headers, in the shared `rlinf-workspace/envs/.uv-python/` directory
instead of relying on the thin image's system Python.

EOS exposes H100 nodes as an exclusive node constraint, not a Slurm GRES.
Submission therefore uses `--constraint=h100 --exclusive` without
`--gpus-per-node`; the in-container preflight fails closed unless exactly eight
GPUs are visible.

The v4 site manifest pins the OCI reference and registry digest, local squashfs
hash, RLinf/Isaac-GR00T/IsaacLab revisions, runtime spec and prepare-script
hashes, model manifest, sanitized USD and task overlay. No Poiesis-owned
checkout, lock file, Python environment or prepare command is part of the
RLinf runtime contract. Git inputs must have clean tracked content and submodules at their
exact revisions. File inputs must match their SHA-256 before submission. Large
immutable inputs live below the persistent EOS workspace `inputs/`; experiment
outputs live below `runs/W73/`.

At allocation start the launcher creates or verifies the shared runtime before
probing its interpreter, Torch, CUDA, FlashAttention, IsaacLab and Ray. Ray is
never started until those checks and the deterministic-seed focused test pass.
The first smoke may spend time materializing the environment; subsequent jobs
reuse it and report the same manifest.

Dependency installation runs in a disposable detached Git worktree, not in the
canonical EOS checkout used for training. The shared-runtime manifest pins the
prepare-script hash and the Git blob inventory for `pyproject.toml` plus
`requirements/`; tickets with unchanged dependency inputs can share the venv,
while interrupted installation cannot leave the training checkout modified.

IsaacLab installs its own qualified Torch, Hydra and NumPy versions while
installing extensions. Runtime preparation therefore disables RLinf's
intermediate FlashAttention build, restores the committed Torch cu128 trio,
pins `hydra-core==1.3.2`, `numpy==1.26.0`, and the PyTorch 2.11-compatible
CPU wheel `torchcodec==0.11.1`, then installs an immutable
FlashAttention 2.8.3 CPython 3.12 wheel built for H100 SM90 against the final
Torch 2.11/cu128 ABI. Both wheel paths and SHA-256 values are part of the site,
runtime spec and generated manifest. This removes the roughly 41-minute source
compile from a fresh allocation while retaining a fail-closed binary
provenance check.
The CPU TorchCodec wheel is intentional: this workload only reaches TorchCodec
through GR00T's dataset-module imports and does not perform GPU video decoding.
It avoids the CUDA 13 runtime dependency of current default Linux wheels.

## Commands

Run these commands on the EOS login node from the persistent RLinf checkout.
They do not require a pre-existing allocation or Remote MCP dock.

```bash
python3 toolkits/eos/start_rlinf.py materialize \
  --template toolkits/eos/gr00t_trocar/site.eos.template.json \
  --output /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W73/site.json

python3 toolkits/eos/start_rlinf.py validate \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W73/site.json

python3 toolkits/eos/start_rlinf.py submit \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W73/site.json \
  --dry-run

python3 toolkits/eos/start_rlinf.py submit \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W73/site.json
```

Before the convergence attempt, materialize and submit a one-step smoke through
the exact same lifecycle:

```bash
python3 toolkits/eos/start_rlinf.py materialize \
  --template toolkits/eos/gr00t_trocar/site.eos.template.json \
  --output /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W73/site-smoke.json \
  --smoke

python3 toolkits/eos/start_rlinf.py submit \
  --site /lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W73/site-smoke.json
```

The smoke retains the four-hour scheduler upper bound but runs at most one
outer step and enforces a 90-minute workload deadline, releasing the allocation
as soon as the end-to-end gate completes.

The real submit command creates an immutable submission receipt. Slurm creates
an attempt named `W73-before-<job-id>/`, and the allocation coordinator records
the exact site, source, image, command, node, deadline and cleanup result.

## Deadline Policy

Slurm owns a four-hour allocation. The workload requests at most 13,200 seconds
after container startup and is additionally capped at `SLURM_JOB_END_TIME -
600s`. The final ten minutes are reserved for process termination, periodic
checkpoint/log flushing, Ray cleanup and receipts. `max_steps=100000` is only a
non-binding safety ceiling; elapsed wall time controls the experiment.

The runner uses the short `/workspace/w73-<job-id>` TMPDIR because Ray's Unix
socket path cannot use the long attempt path. This is the only routine scratch
outside the persistent attempt directory, and the shell removes it on exit.

## Evidence Boundary

An exit-zero or deadline receipt proves operational health, not convergence.
The result must separately report completed outer steps, reward/return curves,
task success metrics, policy/critic health, checkpoints and fixed-seed
evaluation videos. The convergence site runs an 8-env fixed-reset evaluation
and records its videos every five completed outer steps. Evaluation is part of
the four-hour time-to-learning budget but is excluded from steady training-step
timing. If the four-hour result is inconclusive, retain it and change the next
experiment explicitly.
