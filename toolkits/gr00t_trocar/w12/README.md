# RTX PRO 6000 GR00T N1.7 contract

This directory qualifies the shared runtime and workload for the RTX PRO 6000
experiment series. It does not contain retained W13-W16 measurements.

The contract intentionally separates three authorities:

- the Git-tree-attested RLinf integration source;
- the immutable W96-derived model, Isaac-GR00T, Python, TensorRT and asset
  bundle; and
- the W02-qualified offline Healthcare asset resolver, generated from the
  immutable W96 IsaacLab `assets.py` and hash-bound separately; and
- the RTX 6000 host-injected CUDA/Vulkan driver stack.

The host currently exposes the kernel/CUDA driver but not the Vulkan ICD
userspace closure. The launcher therefore mounts an immutable, exact-version
595.58.03 graphics bundle at `/w12-driver`. `libcuda` must still resolve from
the container-runtime host injection; only the NVIDIA GLX/Vulkan library may
resolve from this owned bundle.

`rtx6000_runtime_probe.py` runs before Isaac or Ray. It reuses the audited W96
module, shared-library and ctypes Vulkan checks while requiring exactly eight
RTX PRO 6000 Blackwell Server Edition GPUs with compute capability 12.0.

`rtx6000_qualification_launcher.py` runs the finite qualification in two
phases. `q1` proves the injected CUDA/Vulkan runtime and immutable Python
origins. `q2` repeats that gate, executes one eager true-B8 policy call, and
runs one reset/step with both one and eight Vulkan/PhysX environments. Every
phase requires a new run directory, disables container networking, records the
exact Docker argv, requires the W02 offline resolver SHA
`71f7d05805cd18066f1f93fe3073cf716f590c5bf8ee93f4213b48cbc11647bb`,
and fails unless the named container is removed. Isaac native shutdown may
terminate before the inner smoke promotes `pending_cleanup` to `passed`; the
outer launcher accepts that state only after validating every Env semantic
field and independently proving container removal.

Later performance work uses one source revision and configuration-only arms.
The W13 feature A/B keeps eager execution fixed. W14 qualifies eager, runtime
`torch.compile` and native SM120 TensorRT artifacts. W15 integrates only W14
qualified executors. W16 profiles separate attempts; profiler-perturbed samples
never enter clean performance summaries.
