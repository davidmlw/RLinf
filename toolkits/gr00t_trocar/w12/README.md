# RTX PRO 6000 GR00T N1.7 contract

This directory qualifies the shared runtime and workload for the RTX PRO 6000
experiment series. It does not contain retained W13-W16 measurements.

The contract intentionally separates three authorities:

- the Git-tree-attested RLinf integration source;
- the immutable W96-derived model, Isaac-GR00T, Python, TensorRT and asset
  bundle; and
- the RTX 6000 host-injected CUDA/Vulkan driver stack.

`rtx6000_runtime_probe.py` runs before Isaac or Ray. It reuses the audited W96
module, shared-library and ctypes Vulkan checks while requiring exactly eight
RTX PRO 6000 Blackwell Server Edition GPUs with compute capability 12.0.

Later performance work uses one source revision and configuration-only arms.
The W13 feature A/B keeps eager execution fixed. W14 qualifies eager, runtime
`torch.compile` and native SM120 TensorRT artifacts. W15 integrates only W14
qualified executors. W16 profiles separate attempts; profiler-perturbed samples
never enter clean performance summaries.
