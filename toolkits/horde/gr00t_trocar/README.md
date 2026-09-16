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
