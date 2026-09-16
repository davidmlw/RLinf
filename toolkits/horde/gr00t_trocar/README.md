# GR00T TensorRT on Horde

This directory qualifies the user-owned GR00T N1.7 TensorRT runtime on the
single-GPU Horde host. It does not claim parity with the eight-GPU Trocar
workload.

The runtime probe requires the reviewed PyTorch cu128 and TensorRT 10.15.1.29
environment. It executes a real SM120 BF16 matmul, builds/deserializes/executes
a minimal TensorRT engine, and enumerates the NVIDIA device through the Vulkan
loader before any Isaac environment is started:

```bash
VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
python toolkits/horde/gr00t_trocar/runtime_probe.py \
  --output runs/W03/runtime-probe.json
```

TensorRT plans from SM89 L20 or SM90 H100 hosts are evidence only. Production
ViT and LLM plans must be rebuilt and qualified on the SM120 target.

## Native Rollout and Env runtime

W04 extends the standalone model qualification with a native Isaac Sim runtime.
Docker is not used on Horde. The runtime lives outside the source tree at
`/home/horde/rlinf-workspace/runtime/venvs/isaacsim-6.0-native`; the exact
package and source contract is recorded in `native-runtime-spec.json`.

The native environment uses the exact IsaacLab 8.0.2 source tree recovered
from the qualified Trocar image recorded in `native-runtime-spec.json`. It also
requires the immutable Healthcare asset cache and local asset resolver
qualified by W02. Keep those inputs separate from writable Isaac shader and
application caches.

After installing the runtime and editable IsaacLab packages, run the bounded
environment gates before starting RLinf:

```bash
toolkits/horde/gr00t_trocar/run_native_env_smoke.sh \
  --run-root /home/horde/rlinf-workspace/runs/W04/env-smoke-1 \
  --num-envs 1 --steps 1 --timeout-seconds 900 \
  --venv /home/horde/rlinf-workspace/runtime/venvs/isaacsim-6.0-native \
  --asset-mirror /home/horde/rlinf-workspace/runtime/assets/W04-healthcare-cache

toolkits/horde/gr00t_trocar/run_native_env_smoke.sh \
  --run-root /home/horde/rlinf-workspace/runs/W04/env-smoke-8 \
  --num-envs 8 --steps 2 --timeout-seconds 900 \
  --venv /home/horde/rlinf-workspace/runtime/venvs/isaacsim-6.0-native \
  --asset-mirror /home/horde/rlinf-workspace/runtime/assets/W04-healthcare-cache
```

The launcher treats the execution receipt and post-run process census as the
authority because Isaac Kit may terminate Python from `SimulationApp.close()`.
The resolver is installed in-process before task modules are imported, so the
pinned IsaacLab checkout is not modified and routine assets do not use the
system temporary directory. These gates intentionally validate tensors and
finite outputs. Retained performance launchers must not perform hashes, cosine
comparisons, or output validation inside their measured resident interval.

The Rollout+Env benchmark composes one genuine B8 policy call with sixteen
physical Vulkan environment steps. It retains one discarded full-path warmup
and five measured decisions by default. Set the durable inputs, then select one
of `eager-eager`, `trt-eager`, `trt-pt2`, or `trt-refit-trt`:

```bash
export W04_NATIVE_VENV=/home/horde/rlinf-workspace/runtime/venvs/isaacsim-6.0-native
export W04_ASSET_MIRROR=/home/horde/rlinf-workspace/runtime/assets/W04-healthcare-cache
export W04_GR00T_SOURCE=/home/horde/rlinf-workspace/source/Isaac-GR00T
export W04_MODEL=/home/horde/rlinf-workspace/runs/W03/true-b8-r1/model-view

toolkits/horde/gr00t_trocar/run_rollout_env_benchmark.sh \
  --run-root /home/horde/rlinf-workspace/runs/W04/W04-6-eager-eager \
  --backend eager-eager --warmup 1 --measured 5
```

Lifecycle work, including model load, TensorRT setup, PT2 compilation and DiT
refit/adoption, is reported separately. The retained loop contains no hashes,
cosine checks or output comparisons. This is a single-process systems boundary,
not a Ray/Trainer or PPO-correctness result.
