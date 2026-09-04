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
  W77_BACKBONE_MODEL_ROOT
  W77_TROCAR_METADATA
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
test -d "$W77_BACKBONE_MODEL_ROOT"
test -f "$W77_TROCAR_METADATA"
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
export W77_BACKBONE_MODEL_ROOT
export W77_TROCAR_METADATA
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

w81_trt_overlay="/lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/envs/overlays/tensorrt-10.15.1.29-py312"
w81_trt_engines="/lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W80/g2-export-build-r1-5967165/engines"
case ":${W73_PYTHON_DEPS:-}:" in
  *":$w81_trt_overlay:"*) ;;
  *)
    printf 'W81 TensorRT overlay is absent from W73_PYTHON_DEPS\n' >&2
    exit 2
    ;;
esac
"$W73_RUNTIME_PYTHON" \
  "$W73_SOURCE_ROOT/toolkits/eos/gr00t_trocar/tensorrt/prepare_runtime_overlay.py" \
  verify --output "$w81_trt_overlay" \
  >"$W73_ATTEMPT_ROOT/runtime/tensorrt-overlay-verify.json"
"$W73_RUNTIME_PYTHON" -c \
  'import importlib.metadata as m,json,tensorrt as trt,torch; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"tensorrt_module":trt.__version__,"tensorrt_distribution":m.version("tensorrt-cu12"),"tensorrt_path":trt.__file__},sort_keys=True))' \
  >"$W73_ATTEMPT_ROOT/runtime/tensorrt-import.json"
sha256sum \
  "$w81_trt_engines/rlinf-engine-receipt.json" \
  "$w81_trt_engines/export_metadata.json" \
  "$w81_trt_engines/vit.engine" \
  "$w81_trt_engines/llm_bf16.engine" \
  >"$W73_ATTEMPT_ROOT/runtime/tensorrt-artifacts.sha256"

w83_dit_root="/lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace/runs/W83/W83-refittable-dit-build-r3-5972556"
case "${W83_TRT_DIT_DIAGNOSTIC:-0}" in
  0) ;;
  1)
    sha256sum \
      "$w83_dit_root/engine/rlinf-refittable-dit-engine-receipt.json" \
      "$w83_dit_root/refittable-dit-parameter-map.json" \
      "$w83_dit_root/engine/dit_bf16_refit.engine" \
      >"$W73_ATTEMPT_ROOT/runtime/w83-refittable-dit-artifacts.sha256"
    ;;
  *)
    printf 'W83_TRT_DIT_DIAGNOSTIC must be 0 or 1\n' >&2
    exit 2
    ;;
esac

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
  rollout.model.backbone_model_path="$W77_BACKBONE_MODEL_ROOT"
  actor.model.model_path="$W73_MODEL_ROOT"
  actor.model.backbone_model_path="$W77_BACKBONE_MODEL_ROOT"
)
if [[ -n "$W73_RESUME_DIR" ]]; then
  overrides+=(runner.resume_dir="$W73_RESUME_DIR")
fi
if [[ "${W83_TRT_DIT_DIAGNOSTIC:-0}" == 1 ]]; then
  overrides+=(
    runner.logger.experiment_name=w83_n1d7_trt_dit_identity
    ++rollout.model.tensorrt_dit_diagnostic.enabled=true
    ++rollout.model.tensorrt_dit_diagnostic.engine_path="$w83_dit_root/engine/dit_bf16_refit.engine"
    ++rollout.model.tensorrt_dit_diagnostic.receipt_path="$w83_dit_root/engine/rlinf-refittable-dit-engine-receipt.json"
    ++rollout.model.tensorrt_dit_diagnostic.receipt_sha256=774652d469c47884c6756fe98196884df74cdc129bd611afcf7ea949be7cf024
    ++rollout.model.tensorrt_dit_diagnostic.parameter_map_path="$w83_dit_root/refittable-dit-parameter-map.json"
    ++rollout.model.tensorrt_dit_diagnostic.parameter_map_sha256=df7c72b90629ff6343f52c066a03d430728cc7cd605d12c1b884851fed48c935
    ++rollout.model.tensorrt_dit_diagnostic.source_digest_revision_0=dcadd3c8a2bf405e53dc23aded49c536d0315f4d68f86417feb59a321bd2aaca
    ++rollout.model.tensorrt_dit_diagnostic.revision=0
    ++rollout.model.tensorrt_dit_diagnostic.runtime_version=10.15.1.29
    ++rollout.model.tensorrt_dit_diagnostic.runtime_distribution=tensorrt-cu12
    ++rollout.model.tensorrt_dit_diagnostic.compute_capability='[9,0]'
  )
  printf 'W83_TRT_DIT_DIAGNOSTIC=1\n'
fi
case "${W81_DISABLE_PRE_UPDATE_IDENTITY_GATE:-0}" in
  0) ;;
  1)
    overrides+=(actor.pre_update_same_revision_gate.enabled=false)
    printf 'W81_DISABLE_PRE_UPDATE_IDENTITY_GATE=1\n'
    ;;
  *)
    printf 'W81_DISABLE_PRE_UPDATE_IDENTITY_GATE must be 0 or 1\n' >&2
    exit 2
    ;;
esac
printf 'W73_NEWTON_NUM_SUBSTEPS=%s\n' "$W73_NEWTON_NUM_SUBSTEPS"
cd "$W73_SOURCE_ROOT"
"$W73_RUNTIME_PYTHON" examples/embodiment/train_embodied_agent.py \
  --config-path "$config_dir" \
  --config-name "$config_name" \
  "${overrides[@]}"
