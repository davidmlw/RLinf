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
container=$(printf 'w88-%s' "$run_id" | tr '[:upper:]_' '[:lower:]-')
output="$W88_RUN_ROOT/output"
config_in_container=/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar_prod.yaml
extension_in_container=/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/extension.py
assets_in_container=/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py

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
cp "$W88_CONFIG" "$W88_RUN_ROOT/config.yaml"
cp "$0" "$W88_RUN_ROOT/launcher.sh"
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
  -e PYTHONPATH=/w88-overlay:/workspace/gr00t-n17:/workspace/rlinf-src \
  -v "$W88_SOURCE_ROOT:/workspace/rlinf-src:ro" \
  -v "$W88_GROOT_ROOT:/workspace/gr00t-n17:ro" \
  -v "$W88_PYTHON_OVERLAY:/w88-overlay:ro" \
  -v "$W88_MODEL_INPUT_ROOT:$W88_MODEL_INPUT_ROOT:ro" \
  -v "$W88_MODEL_VIEW:/models/GR00T-N1.7-3B:ro" \
  -v "$W88_ASSET_CACHE:/tmp/Assets" \
  -v "$output:/workspace/isaaclab/output" \
  -v "$W88_CONFIG:$config_in_container:ro" \
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
