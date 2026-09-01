#!/usr/bin/env bash
set -euo pipefail

required=(
  W73_ATTEMPT_ROOT
  W73_SOURCE_ROOT
  W73_CONFIG
  W73_RUNTIME_PYTHON
  W73_ISAACLAB_ROOT
  W73_GROOT_ROOT
  W73_MODEL_ROOT
  W73_HF_CACHE
  W73_TASK_OVERLAY_ROOT
  W73_SANITIZED_TRAY_USD
  W73_HEALTHCARE_ASSETS_ARCHIVE
  W73_MAX_STEPS
  W73_VAL_CHECK_INTERVAL
  W73_SAVE_INTERVAL
  W73_NEWTON_NUM_SUBSTEPS
  W73_DEBUG_NONFINITE
  W73_DEADLINE_UNIX_S
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'missing required environment variable: %s\n' "$name" >&2
    exit 2
  fi
done
if [[ -z "${W73_RESUME_DIR+x}" ]]; then
  printf 'missing required environment variable: W73_RESUME_DIR\n' >&2
  exit 2
fi

test -x "$W73_RUNTIME_PYTHON"
test -d "$W73_SOURCE_ROOT/.git" || test -f "$W73_SOURCE_ROOT/.git"
test -f "$W73_CONFIG"
test -d "$W73_ISAACLAB_ROOT"
test -d "$W73_GROOT_ROOT"
test -d "$W73_MODEL_ROOT"
test -d "$W73_TASK_OVERLAY_ROOT"
test -f "$W73_SANITIZED_TRAY_USD"
test -f "$W73_HEALTHCARE_ASSETS_ARCHIVE"

mkdir -p \
  "$W73_ATTEMPT_ROOT/output" \
  "$W73_ATTEMPT_ROOT/gpu" \
  "$W73_ATTEMPT_ROOT/runtime" \
  "$W73_HF_CACHE"

# Ray's AF_UNIX socket limit makes an attempt-owned Lustre TMPDIR unsafe. Keep
# this forced short-path use isolated and remove it during shell cleanup.
short_tmp="/workspace/w73-${SLURM_JOB_ID:-manual}"
mkdir -p "$short_tmp"
sampler_pid=
cleanup() {
  rc=$?
  set +e
  touch "$W73_ATTEMPT_ROOT/gpu/sampler.stop"
  if [[ -n "$sampler_pid" ]]; then
    wait "$sampler_pid" 2>/dev/null
  fi
  rm -rf "$short_tmp"
  exit "$rc"
}
trap cleanup EXIT

tar -xf "$W73_HEALTHCARE_ASSETS_ARCHIVE" -C "$short_tmp"
test -f \
  "$short_tmp/Assets/Isaac/Healthcare/0.5.0/132c82d/Robots/UnitreeG1/g1_29dof_with_dex3_base_fix/g1_29dof_with_dex3_base_fix.usd"
test -f \
  "$short_tmp/Assets/Isaac/Healthcare/0.5.0/132c82d/Props/LightWheel/Assets/DisposableLaparoscopicPunctureDevice001/DisposableLaparoscopicPunctureDevice005-xform.usd"

python_paths=(
  "$W73_GROOT_ROOT"
  "$W73_SOURCE_ROOT"
  "$W73_TASK_OVERLAY_ROOT"
  "$W73_ISAACLAB_ROOT/source"
)
if [[ -n "${W73_PYTHON_DEPS:-}" ]]; then
  python_paths=("$W73_PYTHON_DEPS" "${python_paths[@]}")
fi
export PYTHONPATH="$(IFS=:; printf '%s' "${python_paths[*]}")"
export TMPDIR="$short_tmp"
export HF_HOME="$W73_HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RLINF_CODE_WORKING_DIR=0
export RLINF_EXT_MODULE=w68_rlinf_extension
export RLINF_CONFIG_FILE="$W73_CONFIG"
export RLINF_DEBUG_NONFINITE="$W73_DEBUG_NONFINITE"
export W68_NEWTON_NUM_SUBSTEPS="$W73_NEWTON_NUM_SUBSTEPS"
export W68_ISAACLAB_SOURCE_ROOT="$W73_ISAACLAB_ROOT/source"
export W68_OVERLAY_ROOT="$W73_TASK_OVERLAY_ROOT"
export W68_SANITIZED_TRAY_USD="$W73_SANITIZED_TRAY_USD"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

{
  printf 'timestamp,index,memory_used_mib,utilization_gpu_pct,power_w\n'
  while [[ ! -f "$W73_ATTEMPT_ROOT/gpu/sampler.stop" ]]; do
    timestamp=$(date -Is)
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits | sed "s/^/$timestamp,/"
    sleep 1
  done
} >"$W73_ATTEMPT_ROOT/gpu/samples.csv" \
  2>"$W73_ATTEMPT_ROOT/gpu/sampler.err" &
sampler_pid=$!

config_dir=$(dirname "$W73_CONFIG")
config_name=$(basename "$W73_CONFIG" .yaml)
overrides=(
  runner.max_epochs="$W73_MAX_STEPS"
  runner.val_check_interval="$W73_VAL_CHECK_INTERVAL"
  runner.save_interval="$W73_SAVE_INTERVAL"
  runner.logger.log_path="$W73_ATTEMPT_ROOT/output"
  runner.debug_nonfinite="$W73_DEBUG_NONFINITE"
  env.train.video_cfg.video_base_dir="$W73_ATTEMPT_ROOT/output/video/train"
  env.eval.video_cfg.video_base_dir="$W73_ATTEMPT_ROOT/output/video/eval"
  rollout.model.model_path="$W73_MODEL_ROOT"
  actor.model.model_path="$W73_MODEL_ROOT"
)
if [[ -n "$W73_RESUME_DIR" ]]; then
  overrides+=(runner.resume_dir="$W73_RESUME_DIR")
fi
printf 'W73_NEWTON_NUM_SUBSTEPS=%s\n' "$W73_NEWTON_NUM_SUBSTEPS"
cd "$W73_SOURCE_ROOT"
"$W73_RUNTIME_PYTHON" examples/embodiment/train_embodied_agent.py \
  --config-path "$config_dir" \
  --config-name "$config_name" \
  "${overrides[@]}"
