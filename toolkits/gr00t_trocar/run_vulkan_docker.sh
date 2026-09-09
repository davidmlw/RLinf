#!/usr/bin/env bash
set -euo pipefail

required=(
  W88_RUN_ROOT
  W88_DOCKER
  W88_IMAGE
  W88_SOURCE_ROOT
  W88_CONFIG
  W88_MODEL_VIEW
  W88_MODEL_INPUT_ROOT
  W88_GROOT_ROOT
  W88_PYTHON_OVERLAY
  W88_TROCAR_METADATA
  W88_ASSET_CACHE
  W88_VULKAN_BASE_EXTENSION
  W88_VULKAN_ASSETS
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'missing required environment variable: %s\n' "$name" >&2
    exit 2
  fi
done

max_epochs=${W88_MAX_EPOCHS:-1}
num_envs=${W88_NUM_ENVS:-64}
run_id=${W88_RUN_ID:-$(basename "$W88_RUN_ROOT")}
arm=${W88_ARM:-control}
container=$(printf 'w88-%s' "$run_id" | tr '[:upper:]_' '[:lower:]-')
output="$W88_RUN_ROOT/output"
config_in_container=/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar_prod.yaml
extension_in_container=/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/extension.py
assets_in_container=/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py
python_path=/w88-overlay:/workspace/gr00t-n17:/workspace/rlinf-src
docker_args=()
mkdir -p "$W88_RUN_ROOT"

case "$arm" in
  control)
    ;;
  a|b2)
    trt_required=(
      W88_TRT_RUNTIME_OVERLAY
      W88_TRT_ENGINE_DIR
      W88_TRT_ENGINE_RECEIPT_SHA256
    )
    if [[ "$arm" == b2 ]]; then
      trt_required+=(
        W88_TRT_DIT_ROOT
        W88_TRT_DIT_QUALIFICATION
      )
    fi
    for name in "${trt_required[@]}"; do
      if [[ -z "${!name:-}" ]]; then
        printf 'missing required environment variable for arm %s: %s\n' \
          "$arm" "$name" >&2
        exit 2
      fi
    done
    test -d "$W88_TRT_RUNTIME_OVERLAY"
    test -d "$W88_TRT_ENGINE_DIR"
    test -f "$W88_TRT_ENGINE_DIR/rlinf-engine-receipt.json"
    printf '%s  %s\n' \
      "$W88_TRT_ENGINE_RECEIPT_SHA256" \
      "$W88_TRT_ENGINE_DIR/rlinf-engine-receipt.json" \
      | sha256sum --check --status
    python_path=/w88-trt-overlay:$python_path
    docker_args+=(
      -e RLINF_GROOT_TRT_ENGINE_DIR=/w88-trt-engine
      -e RLINF_GROOT_TRT_ENGINE_RECEIPT_SHA256="$W88_TRT_ENGINE_RECEIPT_SHA256"
      -v "$W88_TRT_RUNTIME_OVERLAY:/w88-trt-overlay:ro"
      -v "$W88_TRT_ENGINE_DIR:/w88-trt-engine:ro"
    )
    ;;
  *)
    printf 'W88_ARM must be control, a, or b2: %s\n' "$arm" >&2
    exit 2
    ;;
esac

if [[ "$arm" == b2 ]]; then
  test -f "$W88_TRT_DIT_QUALIFICATION"
  bundle="$W88_RUN_ROOT/refittable-dit-bundle.json"
  python3 \
    "$W88_SOURCE_ROOT/toolkits/eos/gr00t_trocar/tensorrt/verify_refittable_dit_bundle.py" \
    --build-root "$W88_TRT_DIT_ROOT" \
    --qualification "$W88_TRT_DIT_QUALIFICATION" \
    --output "$bundle" >/dev/null
  mapfile -t dit_values < <(
    python3 -c \
      'import json,sys; x=json.load(open(sys.argv[1])); print(x["sha256"]["engine_receipt"]); print(x["sha256"]["parameter_map"]); print(x["source_digest_revision_0"])' \
      "$bundle"
  )
  docker_args+=(
    -v "$W88_TRT_DIT_ROOT:/w88-trt-dit:ro"
  )
fi

resolved_config="$W88_RUN_ROOT/config.yaml"
W88_BASE_CONFIG="$W88_CONFIG" \
W88_RESOLVED_CONFIG="$resolved_config" \
W88_SELECTED_ARM="$arm" \
W88_BACKBONE_RECEIPT_SHA256="${W88_TRT_ENGINE_RECEIPT_SHA256:-}" \
W88_DIT_RECEIPT_SHA256="${dit_values[0]:-}" \
W88_DIT_PARAMETER_MAP_SHA256="${dit_values[1]:-}" \
W88_DIT_SOURCE_DIGEST="${dit_values[2]:-}" \
python3 - <<'PY'
import os
from pathlib import Path

import yaml

source = Path(os.environ["W88_BASE_CONFIG"])
target = Path(os.environ["W88_RESOLVED_CONFIG"])
arm = os.environ["W88_SELECTED_ARM"]
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["runner"]["logger"]["experiment_name"] = f"w88_n1d7_vulkan_{arm}"

if arm in {"a", "b2"}:
    config["rollout"]["model"]["tensorrt_backbone"] = {
        "enabled": True,
        "engine_dir": "/w88-trt-engine",
        "receipt_path": "/w88-trt-engine/rlinf-engine-receipt.json",
        "receipt_sha256": os.environ["W88_BACKBONE_RECEIPT_SHA256"],
        "static_batch_size": 8,
        "sequence_opt": 208,
        "runtime_version": "10.15.1.29",
        "runtime_distribution": "tensorrt-cu12",
        "compute_capability": [8, 9],
    }

if arm == "b2":
    config["rollout"]["model"]["tensorrt_dit"] = {
        "enabled": True,
        "engine_path": "/w88-trt-dit/engine/dit_bf16_refit.engine",
        "receipt_path": (
            "/w88-trt-dit/engine/rlinf-refittable-dit-engine-receipt.json"
        ),
        "receipt_sha256": os.environ["W88_DIT_RECEIPT_SHA256"],
        "parameter_map_path": "/w88-trt-dit/refittable-dit-parameter-map.json",
        "parameter_map_sha256": os.environ["W88_DIT_PARAMETER_MAP_SHA256"],
        "source_digest_revision_0": os.environ["W88_DIT_SOURCE_DIGEST"],
        "revision": 0,
        "runtime_version": "10.15.1.29",
        "runtime_distribution": "tensorrt-cu12",
        "compute_capability": [8, 9],
        "online_refit": True,
        "lineage_receipt_mode": "gpu_transform_validation",
        "probe_each_revision": True,
        "minimum_probe_cosine": 0.999,
        "maximum_probe_relative_l2": 0.05,
        "minimum_free_device_bytes": 4294967296,
        "ppo_authority_status": "failed_ratio_kl_approximate_behavior_only",
        "shadow_eager": False,
    }
    config["actor"]["pre_update_same_revision_gate"]["enabled"] = False

target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

if [[ "$(git -C "$W88_SOURCE_ROOT" status --short)" ]]; then
  printf 'RLinf source must be clean\n' >&2
  exit 2
fi
for path in \
  "$W88_DOCKER" "$W88_CONFIG" "$W88_TROCAR_METADATA" \
  "$W88_VULKAN_BASE_EXTENSION" "$W88_VULKAN_ASSETS"; do
  if [[ ! -f "$path" ]]; then
    printf 'required file is missing: %s\n' "$path" >&2
    exit 2
  fi
done
for path in \
  "$W88_SOURCE_ROOT" "$W88_MODEL_VIEW" "$W88_MODEL_INPUT_ROOT" \
  "$W88_GROOT_ROOT" "$W88_PYTHON_OVERLAY" "$W88_ASSET_CACHE"; do
  if [[ ! -d "$path" ]]; then
    printf 'required directory is missing: %s\n' "$path" >&2
    exit 2
  fi
done

mkdir -p "$W88_RUN_ROOT" "$output" "$W88_RUN_ROOT/gpu"
chmod 0777 "$W88_RUN_ROOT" "$output" "$W88_RUN_ROOT/gpu"
cp "$0" "$W88_RUN_ROOT/launcher.sh"
printf '%s\n' "$arm" >"$W88_RUN_ROOT/arm"
git -C "$W88_SOURCE_ROOT" rev-parse HEAD >"$W88_RUN_ROOT/source.sha"
"$W88_DOCKER" image inspect "$W88_IMAGE" >"$W88_RUN_ROOT/image-inspect.json"
sha256sum \
  "$W88_RUN_ROOT/config.yaml" \
  "$W88_VULKAN_BASE_EXTENSION" \
  "$W88_VULKAN_ASSETS" \
  >"$W88_RUN_ROOT/runtime-inputs.sha256"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'GPU preflight failed: compute processes are already running\n' >&2
  exit 2
fi

sampler_pid=
cleanup() {
  rc=$?
  set +e
  touch "$W88_RUN_ROOT/gpu/stop"
  if [[ -n "$sampler_pid" ]]; then
    wait "$sampler_pid" 2>/dev/null
  fi
  if "$W88_DOCKER" inspect "$container" >/dev/null 2>&1; then
    "$W88_DOCKER" inspect "$container" >"$W88_RUN_ROOT/container-inspect.json"
    "$W88_DOCKER" logs "$container" >"$W88_RUN_ROOT/container.log" 2>&1
    "$W88_DOCKER" rm -f "$container" >/dev/null 2>&1
  fi
  printf '%s\n' "$rc" >"$W88_RUN_ROOT/exit"
  trap - EXIT
  exit "$rc"
}
trap cleanup EXIT INT TERM

printf 'timestamp,index,memory_used_mib,utilization_gpu_pct,power_w\n' \
  >"$W88_RUN_ROOT/gpu/samples.csv"
(
  while [[ ! -f "$W88_RUN_ROOT/gpu/stop" ]]; do
    timestamp=$(date -Is)
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits | sed "s/^/$timestamp,/"
    sleep 1
  done
) >>"$W88_RUN_ROOT/gpu/samples.csv" 2>"$W88_RUN_ROOT/gpu/sampler.err" &
sampler_pid=$!

"$W88_DOCKER" rm -f "$container" >/dev/null 2>&1 || true
"$W88_DOCKER" run --name "$container" \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --network host \
  --entrypoint /bin/bash \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  "${docker_args[@]}" \
  -e OMNI_KIT_ACCEPT_EULA=yes \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e NO_ALBUMENTATIONS_UPDATE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RAY_DEDUP_LOGS=0 \
  -e RLINF_CODE_WORKING_DIR=0 \
  -e RLINF_EXT_MODULE=toolkits.gr00t_trocar.vulkan_extension \
  -e RLINF_CONFIG_FILE="$config_in_container" \
  -e W77_BACKBONE_MODEL_ROOT="$W88_MODEL_INPUT_ROOT/Cosmos-Reason2-2B" \
  -e W77_TROCAR_METADATA="$W88_TROCAR_METADATA" \
  -e PYTHONPATH="$python_path" \
  -v "$W88_SOURCE_ROOT:/workspace/rlinf-src:ro" \
  -v "$W88_GROOT_ROOT:/workspace/gr00t-n17:ro" \
  -v "$W88_PYTHON_OVERLAY:/w88-overlay:ro" \
  -v "$W88_MODEL_INPUT_ROOT:$W88_MODEL_INPUT_ROOT:ro" \
  -v "$W88_MODEL_VIEW:/models/GR00T-N1.7-3B:ro" \
  -v "$W88_ASSET_CACHE:/tmp/Assets" \
  -v "$output:/workspace/isaaclab/output" \
  -v "$resolved_config:$config_in_container:ro" \
  -v "$W88_VULKAN_BASE_EXTENSION:$extension_in_container:ro" \
  -v "$W88_VULKAN_ASSETS:$assets_in_container:ro" \
  "$W88_IMAGE" -lc "
set -euo pipefail
cd /workspace/isaaclab
\$PY - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import rlinf

print('W88_RUNTIME', {
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'rlinf': rlinf.__file__,
    'isaaclab': metadata.version('isaaclab'),
}, flush=True)
assert rlinf.__file__.startswith('/workspace/rlinf-src/')
PY
./isaaclab.sh train \
  --rl_library rlinf \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_prod \
  --model_path /models/GR00T-N1.7-3B \
  --num_envs $num_envs \
  --max_epochs $max_epochs \
  2>&1 | tee /workspace/isaaclab/output/bench.log
"
