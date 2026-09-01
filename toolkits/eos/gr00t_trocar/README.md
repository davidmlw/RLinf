# EOS GR00T Trocar Baseline

This directory owns the reproducible W71-before control on EOS H100. It uses
RLinf base `0f9ea98c`, GR00T N1.5, the Newton/MJWarp Trocar task, and the
production action-chunk-16 PPO configuration. Rollout backbone feature reuse
and the other local performance features are absent.

## Runtime

The frozen OCI input is:

```text
gitlab-master.nvidia.com:5005/liweim/image/poiesis:be095dc
registry digest: sha256:aab301569e09f60e143ddcb7749b2610fe522503edcdc8538efbb4a446b1a53f
squashfs sha256: 64bbd7bda0f8d65d298073377a3e2331e91a75c49d459893ae5b3096410b022c
```

The image alone is not the complete runtime. The site manifest pins the RLinf,
Poiesis, Isaac-GR00T and IsaacLab revisions, model directory, compatibility
Python files, sanitized USD and task overlay. Large immutable inputs live below
the persistent EOS workspace `inputs/`; experiment outputs live below
`runs/W73/`.

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
evaluation videos. If the four-hour result is inconclusive, retain it and
change the next experiment explicitly.
