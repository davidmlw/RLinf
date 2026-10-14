# RTX eager learning reference

This independent Kiln-owned reference starts at RLInf commit
`6d0db56bf26f972cd27fa29535f5eb939e80e5bf` (L20 W85). Its native GR00T model,
flow-SDE noise 0.5, PPO, metrics and synchronization remain RLInf-owned code.
TensorRT, feature reuse, compiled graphs and head-only capsules are disabled.
The RLInf team's existing local/remote checkouts are not modified.

The RTX simulator contract is Lab3 / Isaac Sim 6.0.1.0 / Torch2.11 / Python3.12,
whereas native model/Ray workers retain Torch2.7 / Python3.11. An optional
`env.*.external_simulator` binding replaces only the existing simulator
subprocess. Physical step/reset results cross a private Unix socket as explicit
dtype/shape/CPU bytes, not pickled CUDA storage or Python functions. The original
RLInf environment wrapper retains chunking, reward deltas, return/success
metrics and terminal handling. Lab3 xyzw quaternions are converted once to the
wrapper's original wxyz input ABI. The two upstream cameras, task assets,
randomization and reward weight20 match the qualified RTX task. No Poiesis
model, RL, transport or control-plane code is imported.

The CPU copies are a compatibility cost. This is not a performance comparison
against either W85 or Poiesis W60. Sim6 versus Lab2, flow noise0.5 versus0.3,
full-model versus head-only FSDP and evaluator sampling remain explicit deltas.

`prepare.py` creates a UV overlay below `_build/rlinf-reference`. It inherits
only qualified third-party packages (Torch/FlashAttention/NCCL/GR00T and their
dependencies) read-only from the existing model prefix, adds pinned Ray/runtime
dependencies, and never installs into either existing Poiesis environment.
Record both dependency prefixes in the build receipt. The simulator prefix is
also read-only. No package installation happens at worker startup.

All simulator HOME/cache/logs and learning outputs belong to the attempt under
`_runs`; sockets belong below `/tmp/kiln/<allocation>/<attempt>`. Keep clean
fixed source, local commit/push and locked host synchronization. Small CPU
codec tests precede the actual simulator and native RLInf functional gates;
neither is learning qualification. The registered learning witness is one fresh
100-step lineage with fixed evaluations at0/20/50/100 and bounded checkpoints.
