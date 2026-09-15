# RLInf EOS GR00T N1.5 Stack Cube control

This directory supplies the missing RLInf/EOS/OvPhysX+NewtonWarpRenderer arm
for Kiln W43. It is an environment-domain experiment, not a performance arm.

The source baseline is RLInf `0f9ea98c` plus the opt-in deterministic-seed
commit `81857b28`; no Trocar source or diagnostics are carried into this
branch. W43 reuses only W73's separately qualified EOS Python/Torch/IsaacLab/
GR00T runtime artifact. `w43_newton_stack_cube.py` is a standalone RLInf environment adapter
derived from the qualified Poiesis W42 task; it imports no Poiesis runtime.
`w43_rlinf_extension.py` registers only that task and leaves native RLInf
GR00T conversion, rollout, FSDP PPO, weight sync, and metrics in authority.

## Causal boundary

The four observations are:

| Framework | L20 PhysX/RTX/Vulkan | EOS OvPhysX/NewtonWarpRenderer |
| --- | --- | --- |
| RLInf | W85 retained oracle | W43 new arm |
| Poiesis | W38 retained oracle | W42 retained oracle |

The EOS arm still changes hardware, driver and IsaacLab major version. It can
identify whether low policy signal follows the framework or the environment
stack, but cannot by itself rank Newton physics against Vulkan or RTX.

## Gate order

1. Validate `control-contract.json` and source/runtime provenance.
2. Run `W43_MODE=initial-eval` for one isolated 96-episode r0 evaluation.
3. Run `W43_MODE=train` with `W43_MAX_STEPS=1` to obtain the W85-compatible
   training-reset rollout boundary and one finite PPO update.
4. Continue to registered r10/r25 only if r0 has useful sparse-reward support.

Feature reuse, TensorRT, `torch.compile`, async PPO, environment epoch folding,
and reduced camera cadence are intentionally excluded.

The fixed evaluation disables auto-reset: each of 32 environments executes
one complete 448-action horizon in each of three epochs.  After every full
reset, one neutral open-gripper action refreshes OvPhysX/Newton kinematics and
the attached wrist-camera pose; that compatibility step is excluded from the
policy trajectory and mirrors the qualified W42 environment contract.
Training/evaluation environment seeds are `0/42`; their rollout-noise seeds
are `64101/864101`, and the value head is initialized with seed `1234`.
