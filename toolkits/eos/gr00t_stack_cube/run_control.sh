#!/usr/bin/env bash
set -euo pipefail

required=(
  W43_ATTEMPT_ROOT
  W43_SOURCE_ROOT
  W43_CONFIG
  W43_RUNTIME_PYTHON
  W43_ISAACLAB_ROOT
  W43_GROOT_ROOT
  W43_MODEL_ROOT
  W43_HF_CACHE
  W43_MODE
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'missing required environment variable: %s\n' "$name" >&2
    exit 2
  fi
done

case "$W43_MODE" in
  initial-eval|train) ;;
  *)
    printf 'W43_MODE must be initial-eval or train, got %s\n' "$W43_MODE" >&2
    exit 2
    ;;
esac

test -x "$W43_RUNTIME_PYTHON"
test -d "$W43_SOURCE_ROOT/.git" || test -f "$W43_SOURCE_ROOT/.git"
test -f "$W43_CONFIG"
test -d "$W43_ISAACLAB_ROOT"
test -d "$W43_GROOT_ROOT"
test -d "$W43_MODEL_ROOT"

mkdir -p \
  "$W43_ATTEMPT_ROOT/control" \
  "$W43_ATTEMPT_ROOT/gpu" \
  "$W43_ATTEMPT_ROOT/results" \
  "$W43_ATTEMPT_ROOT/output" \
  "$W43_HF_CACHE"

# Ray needs short Unix socket paths. This attempt-owned node-local directory is
# the only W43 state outside the shared run root and is removed on every exit.
allocation_id="${W43_ALLOCATION_ID:-${SLURM_JOB_ID:-manual}}"
short_tmp="/tmp/kiln/$allocation_id/$(basename "$W43_ATTEMPT_ROOT")"
mkdir -p "$short_tmp"
sampler_pid=
cleanup() {
  rc=$?
  set +e
  touch "$W43_ATTEMPT_ROOT/gpu/sampler.stop"
  if [[ -n "$sampler_pid" ]]; then
    wait "$sampler_pid" 2>/dev/null
  fi
  "$W43_RUNTIME_PYTHON" -c \
    'import sys; from ray.scripts.scripts import main; sys.argv[0] = "ray"; raise SystemExit(main())' \
    stop --force \
    >"$W43_ATTEMPT_ROOT/control/ray-stop.out" 2>&1
  rm -rf "$short_tmp"
  exit "$rc"
}
trap cleanup EXIT

overlay_root="$W43_SOURCE_ROOT/toolkits/eos/gr00t_stack_cube"
python_paths=(
  "$W43_GROOT_ROOT"
  "$W43_SOURCE_ROOT"
  "$overlay_root"
  "$W43_ISAACLAB_ROOT/source"
)
if [[ -n "${W43_PYTHON_DEPS:-}" ]]; then
  python_paths=("$W43_PYTHON_DEPS" "${python_paths[@]}")
fi
export PYTHONPATH="$(IFS=:; printf '%s' "${python_paths[*]}")"
export TMPDIR="$short_tmp"
export HF_HOME="$W43_HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RLINF_CODE_WORKING_DIR=0
export RLINF_EXT_MODULE=w43_rlinf_extension
export RLINF_CONFIG_FILE="$W43_CONFIG"
export W43_ISAACLAB_SOURCE_ROOT="$W43_ISAACLAB_ROOT/source"
export W43_ATTEMPT_ROOT
export W43_INITIAL_EVAL_ONLY=false
export EMBODIED_PATH="$W43_SOURCE_ROOT/examples/embodiment"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

if [[ "$W43_MODE" == initial-eval ]]; then
  export W43_INITIAL_EVAL_ONLY=true
  rollout_seed=864101
else
  rollout_seed=64101
fi

{
  printf 'timestamp,index,memory_used_mib,utilization_gpu_pct,power_w\n'
  while [[ ! -f "$W43_ATTEMPT_ROOT/gpu/sampler.stop" ]]; do
    timestamp=$(date -Is)
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits | sed "s/^/$timestamp,/"
    sleep 1
  done
} >"$W43_ATTEMPT_ROOT/gpu/samples.csv" \
  2>"$W43_ATTEMPT_ROOT/gpu/sampler.err" &
sampler_pid=$!

max_steps="${W43_MAX_STEPS:-1}"
if [[ "$W43_MODE" == initial-eval ]]; then
  # Worker construction only enables the eval path when validation is active.
  # The driver extension below replaces the training loop before it can use
  # this interval.
  val_interval=1
else
  val_interval="${W43_VAL_CHECK_INTERVAL:--1}"
fi
save_interval="${W43_SAVE_INTERVAL:-10}"
config_dir=$(dirname "$W43_CONFIG")
config_name=$(basename "$W43_CONFIG" .yaml)
entrypoint="$W43_SOURCE_ROOT/examples/embodiment/train_embodied_agent.py"

cd "$W43_SOURCE_ROOT"
hydra_args=(
  --config-path "$config_dir"
  --config-name "$config_name"
  runner.max_steps="$max_steps"
  runner.val_check_interval="$val_interval"
  runner.save_interval="$save_interval"
  rollout.seed="$rollout_seed"
  actor.model.value_head_init_seed=1234
  runner.logger.log_path="$W43_ATTEMPT_ROOT/output"
  env.train.video_cfg.video_base_dir="$W43_ATTEMPT_ROOT/output/video/train"
  env.eval.video_cfg.video_base_dir="$W43_ATTEMPT_ROOT/output/video/eval"
  rollout.model.model_path="$W43_MODEL_ROOT"
  actor.model.model_path="$W43_MODEL_ROOT"
)
if [[ "$W43_MODE" == initial-eval ]]; then
  # RLINF_EXT_MODULE is loaded automatically inside each Worker, but the
  # initial-evaluation hook changes EmbodiedRunner and therefore must also be
  # installed in the driver process before the normal entrypoint is executed.
  "$W43_RUNTIME_PYTHON" -c '
import runpy
import sys
import w43_rlinf_extension

w43_rlinf_extension.register()
entrypoint = sys.argv[1]
sys.argv = [entrypoint, *sys.argv[2:]]
runpy.run_path(entrypoint, run_name="__main__")
' "$entrypoint" "${hydra_args[@]}"
else
  "$W43_RUNTIME_PYTHON" "$entrypoint" "${hydra_args[@]}"
fi
