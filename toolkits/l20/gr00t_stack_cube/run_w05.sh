#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_w05.sh ATTEMPT MAX_STEPS

Environment overrides:
  SAVE_INTERVAL=10
  VAL_CHECK_INTERVAL=10
  VERIFY_TRAJECTORY=false
  RESUME_DIR=/absolute/path/to/checkpoints/global_step_N
  W05_RUNTIME_ROOT=/home/liweim/rl-workspace/rlinf/main
  W05_ISAAC_ROOT=/home/liweim/rl-workspace/poiesis-runtime/isaac-sim-5.1.0
EOF
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

ATTEMPT=$1
MAX_STEPS=$2
if [[ ! "$ATTEMPT" =~ ^W05-[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ATTEMPT must start with W05- and contain only safe path characters" >&2
    exit 2
fi
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer" >&2
    exit 2
fi

SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUNTIME_ROOT=${W05_RUNTIME_ROOT:-/home/liweim/rl-workspace/rlinf/main}
ISAAC=${W05_ISAAC_ROOT:-/home/liweim/rl-workspace/poiesis-runtime/isaac-sim-5.1.0}
PYTHON=${W05_PYTHON:-$RUNTIME_ROOT/.venv/bin/python}
RAY=${W05_RAY:-$RUNTIME_ROOT/.venv/bin/ray}
SITE=$RUNTIME_ROOT/.venv/lib/python3.11/site-packages
GROOT=$RUNTIME_ROOT/.venv/gr00t
MODEL=${GR00T_STACK_CUBE_MODEL_PATH:-$RUNTIME_ROOT/models/RLinf-Gr00t-SFT-Stack-cube}
RUN_ROOT=$SOURCE/results/W05/$ATTEMPT
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-10}
VERIFY_TRAJECTORY=${VERIFY_TRAJECTORY:-false}
RESUME_DIR=${RESUME_DIR:-}
TMP_BASE=${W05_TMP_ROOT:-$HOME/r/w05}
RAY_TMPDIR=$TMP_BASE/ray
HF_HOME=${HF_HOME:-$HOME/tmp/W05-hf-cache}

for interval in "$SAVE_INTERVAL" "$VAL_CHECK_INTERVAL"; do
    if [[ ! "$interval" =~ ^-?[0-9]+$ ]]; then
        echo "save/eval intervals must be integers: $interval" >&2
        exit 2
    fi
done
if (( SAVE_INTERVAL >= 0 && VAL_CHECK_INTERVAL > 0 && SAVE_INTERVAL % VAL_CHECK_INTERVAL != 0 )); then
    echo "SAVE_INTERVAL must be divisible by VAL_CHECK_INTERVAL" >&2
    exit 2
fi

# Ray appends a long session/sockets suffix and AF_UNIX paths are limited to
# 107 bytes. Keep the configurable prefix short enough to fail before launch.
if (( ${#RAY_TMPDIR} > 40 )); then
    echo "RAY_TMPDIR is too long for Ray Unix sockets: $RAY_TMPDIR" >&2
    exit 1
fi

for path in "$PYTHON" "$RAY" "$ISAAC/setup_conda_env.sh" "$MODEL"; do
    if [[ ! -e "$path" ]]; then
        echo "required W05 input is missing: $path" >&2
        exit 1
    fi
done
if [[ -e "$RUN_ROOT" ]]; then
    echo "refusing to reuse existing run root: $RUN_ROOT" >&2
    exit 1
fi
if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "W05 source worktree must be clean" >&2
    exit 1
fi
if [[ $(nvidia-smi -L | wc -l) -ne 8 ]]; then
    echo "W05 requires exactly eight visible GPUs" >&2
    exit 1
fi
if [[ $(nvidia-smi --query-gpu=name --format=csv,noheader | rg -c '^NVIDIA L20$') -ne 8 ]]; then
    echo "W05 requires eight NVIDIA L20 GPUs" >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | rg -q '[0-9]'; then
    echo "W05 refuses to start while GPU compute processes are active" >&2
    exit 1
fi
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
    echo "W05 refuses to replace an active Ray cluster" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT" "$TMP_BASE" "$HF_HOME"

set +u
# shellcheck source=/dev/null
source "$ISAAC/setup_conda_env.sh"
set -u

export GR00T_STACK_CUBE_MODEL_PATH=$MODEL
export EMBODIED_PATH=$SOURCE/examples/embodiment
export TMPDIR=${W05_GENERAL_TMP_ROOT:-$HOME/tmp}
export RAY_TMPDIR
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_AUTOSCALER_V2=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RLINF_CODE_WORKING_DIR=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMNI_KIT_ACCEPT_EULA=Y
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
PYTHON_BIN_DIR=$(dirname "$PYTHON")
export PATH="/usr/local/cuda-12.6/bin:$PYTHON_BIN_DIR:$HOME/.local/bin:$PATH"
export LD_PRELOAD=$ISAAC/kit/libcarb.so
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$HOME/.local-libs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SOURCE:$GROOT:$RUNTIME_ROOT:$SITE:$ISAAC/python_packages:$ISAAC/exts/isaacsim.simulation_app:$ISAAC/extsDeprecated/omni.isaac.kit:$ISAAC/kit/kernel/py:$ISAAC/kit/plugins/bindings-python:$ISAAC/exts/isaacsim.robot_motion.lula/pip_prebundle:$ISAAC/exts/isaacsim.asset.exporter.urdf/pip_prebundle:$ISAAC/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle:$ISAAC/exts/omni.isaac.core_archive/pip_prebundle:$ISAAC/exts/omni.isaac.ml_archive/pip_prebundle:$ISAAC/exts/omni.pip.compute/pip_prebundle:$ISAAC/exts/omni.pip.cloud/pip_prebundle"

rm -rf "$RAY_TMPDIR"

{
    printf 'attempt=%s\n' "$ATTEMPT"
    printf 'source=%s\n' "$SOURCE"
    printf 'source_sha=%s\n' "$(git -C "$SOURCE" rev-parse HEAD)"
    printf 'source_status_lines=%s\n' "$(git -C "$SOURCE" status --porcelain | wc -l)"
    printf 'python=%s\n' "$PYTHON"
    printf 'isaac=%s\n' "$ISAAC"
    printf 'model=%s\n' "$MODEL"
    printf 'max_steps=%s\n' "$MAX_STEPS"
    printf 'save_interval=%s\n' "$SAVE_INTERVAL"
    printf 'val_check_interval=%s\n' "$VAL_CHECK_INTERVAL"
    printf 'verify_trajectory=%s\n' "$VERIFY_TRAJECTORY"
    printf 'resume_dir=%s\n' "$RESUME_DIR"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} >"$RUN_ROOT/provenance.txt"

"$PYTHON" - <<'PY' >"$RUN_ROOT/runtime.txt"
import importlib.metadata
import importlib.util
import sys

print(f"python={sys.version}")
for name in ("rlinf", "gr00t", "isaaclab", "torch", "ray"):
    spec = importlib.util.find_spec(name)
    print(f"module.{name}={None if spec is None else spec.origin}")
for name in ("gr00t", "isaaclab", "torch", "ray"):
    try:
        print(f"distribution.{name}={importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"distribution.{name}=missing")
PY

CMD=(
    "$PYTHON"
    "$SOURCE/examples/embodiment/train_embodied_agent.py"
    --config-path "$SOURCE/examples/embodiment/config"
    --config-name isaaclab_franka_stack_cube_ppo_gr00t_feature_bundle
    "runner.max_steps=$MAX_STEPS"
    "runner.save_interval=$SAVE_INTERVAL"
    "runner.val_check_interval=$VAL_CHECK_INTERVAL"
    "runner.logger.log_path=$RUN_ROOT/output"
    "runner.logger.experiment_name=$ATTEMPT"
    "rollout.pinned_feature_verify_trajectory=$VERIFY_TRAJECTORY"
)
if [[ -n "$RESUME_DIR" ]]; then
    if [[ ! -d "$RESUME_DIR/actor" ]]; then
        echo "resume checkpoint is missing its actor directory: $RESUME_DIR/actor" >&2
        exit 1
    fi
    CMD+=("runner.resume_dir=$RESUME_DIR")
fi

printf '%q ' "${CMD[@]}" >"$RUN_ROOT/command.txt"
printf '\n' >>"$RUN_ROOT/command.txt"
cp "$SOURCE/examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t_feature_bundle.yaml" "$RUN_ROOT/input-config.yaml"

cleanup() {
    local rc=$?
    set +e
    "$RAY" stop --force >"$RUN_ROOT/ray-stop.out" 2>"$RUN_ROOT/ray-stop.err"
    local ray_rc=$?
    printf '%s\n' "$rc" >"$RUN_ROOT/exit"
    printf '%s\n' "$ray_rc" >"$RUN_ROOT/ray-stop.exit"
    exit "$rc"
}
trap cleanup EXIT

printf 'W05 launch %s\n' "$(date -Is)"
printf 'run_root=%s\n' "$RUN_ROOT"
"${CMD[@]}" > >(tee "$RUN_ROOT/stdout") 2> >(tee "$RUN_ROOT/stderr" >&2)
