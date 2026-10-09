#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  printf 'usage: %s <arm> <gate|perf> <config> <run-root> <max-epochs>\n' "$0" >&2
  exit 2
fi

arm=$1
phase=$2
config=$3
run_root=$4
max_epochs=$5

workspace=/home/scratch.liweim/rl-workspace
authority="$workspace/runtime/W12"
source_root="$workspace/runtime/W15/source"
bundle="$authority/w96-seed/immutable-n17-overlay"
asset_seed="$bundle/assets/assets-cache"
gr00t_source="$bundle/sources/isaac-gr00t"
python_overlay="$bundle/python/w96-overlay"
trt_runtime="$bundle/runtime/tensorrt-10.15.1.29"
model_root="$bundle/model"
extension="$bundle/overrides/extension.py"
assets_override="$authority/immutable/overrides/assets-offline-w02.py"
metadata="$bundle/overrides/trocar-metadata.json"
graphics_runtime="$authority/tools/nvidia-graphics-595.58.03/extracted"
contract="$source_root/toolkits/gr00t_trocar/w15/contract.json"
artifact_root="$workspace/runs/W14/W14-6-rtx6000-b8-executor-matrix/artifacts"
docker=$(command -v docker)
image='chenchaox72877/trocar-rlinf-bench@sha256:9f02e069ccb0e0a7e536833e789666e85f039c9024a77ac2f219c42cb1dfcf01'
image_id='sha256:db746c040dd15cdd68fdcda5b40f514bd2bf31d0fb462ae493fe83b7a0142bf1'
container="w15-$(basename "$run_root" | tr '[:upper:]_' '[:lower:]-')"
container_config=/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar_prod.yaml
container_extension=/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/extension.py
container_assets=/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py

case "$phase:$arm" in
  gate:eager-eager) expected_config_sha=aed9247d2b444af7b7bbbebadf671bab9b9dddcb5614d1eeebeca9b8cb2314a1 ;;
  gate:pt2-pt2) expected_config_sha=97e9b876e922c64009abaab66ee1e8d7fd5445673196395c187308367656c5e7 ;;
  gate:trt-eager) expected_config_sha=189f24c522e5dac6cc220e506e2de64c9844f9ed66a6146a923038e48f666846 ;;
  gate:trt-pt2) expected_config_sha=efe42ccd637d107f7beb0acd4f62b28a0b141e9914fc15cdd7e1dfd9af0fc02b ;;
  gate:trt-refit-trt) expected_config_sha=a3c18ad078653aa5492d8b73c087abb0fc93daaab01cb261cd718f9977a5fc7e ;;
  perf:eager-eager) expected_config_sha=65e9366efd84e805465ece8e5ce437da77ec327c5f4e3fa7e00537a3eee9eeb6 ;;
  perf:pt2-pt2) expected_config_sha=f7855fc748ef57015d8ce7c083b4bdac03e7c2e90ceef2b4b1206c29019850b2 ;;
  perf:trt-eager) expected_config_sha=9bda87512ee6a958649aad5dfb2788c7f69f2d1888d99277f60ce12593e031bf ;;
  perf:trt-pt2) expected_config_sha=bc606a5f16b11a3728b1ad8dc1fb9f96d688c9c4cb7a6553df3b8ccb26b46400 ;;
  perf:trt-refit-trt) expected_config_sha=15f8cd5dd7524ee950d86a716e5888bb6c25997f5a37149839cbc766cf558f41 ;;
  *)
    printf 'unsupported W15 phase/arm: %s/%s\n' "$phase" "$arm" >&2
    exit 2
    ;;
esac
if [[ ! "$max_epochs" =~ ^[1-9][0-9]*$ ]]; then
  printf 'max-epochs must be a positive integer: %s\n' "$max_epochs" >&2
  exit 2
fi
if [[ ( "$phase" == gate && "$max_epochs" -ne 1 ) || \
  ( "$phase" == perf && "$max_epochs" -ne 5 ) ]]; then
  printf 'W15 phase/max-epochs mismatch: %s/%s\n' "$phase" "$max_epochs" >&2
  exit 2
fi
if [[ -e "$run_root" ]]; then
  printf 'run root must be new and absent: %s\n' "$run_root" >&2
  exit 2
fi

for path in "$source_root" "$asset_seed" "$gr00t_source" "$python_overlay" \
  "$trt_runtime" "$model_root/GR00T-N1.7-3B" \
  "$model_root/Cosmos-Reason2-2B" "$graphics_runtime" "$artifact_root"; do
  [[ -d "$path" ]] || { printf 'missing required directory: %s\n' "$path" >&2; exit 2; }
done
for path in "$contract" "$config" "$docker" "$extension" \
  "$assets_override" "$metadata"; do
  [[ -f "$path" ]] || { printf 'missing required file: %s\n' "$path" >&2; exit 2; }
done
actual_config_sha=$(sha256sum "$config" | awk '{print $1}')
if [[ "$actual_config_sha" != "$expected_config_sha" ]]; then
  printf 'config SHA mismatch: expected=%s actual=%s\n' \
    "$expected_config_sha" "$actual_config_sha" >&2
  exit 2
fi

mkdir -p "$run_root/receipts" "$run_root/output" "$run_root/gpu" \
  "$run_root/scratch/home" "$run_root/scratch/tmp" \
  "$run_root/scratch/cache" "$run_root/scratch/hf" \
  "$run_root/scratch/ray" "$run_root/scratch/torchinductor" \
  "$run_root/scratch/kit-cache" "$run_root/scratch/kit-data" \
  "$run_root/scratch/rlinf-entry-logs"

actual_image_id=$("$docker" image inspect --format '{{.Id}}' "$image")
if [[ "$actual_image_id" != "$image_id" ]]; then
  printf 'container image ID mismatch: expected=%s actual=%s\n' \
    "$image_id" "$actual_image_id" >&2
  exit 2
fi

# shellcheck disable=SC2329  # Invoked through cleanup traps.
inspect_container() {
  local label=$1
  local stdout_path="$run_root/receipts/container-${label}-inspect.stdout"
  local stderr_path="$run_root/receipts/container-${label}-inspect.stderr"
  local rc_path="$run_root/receipts/container-${label}-inspect.rc"
  local state_path="$run_root/receipts/container-${label}-inspect.state"
  local rc error output
  if "$docker" inspect --type container "$container" >"$stdout_path" 2>"$stderr_path"; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" >"$rc_path"
  error=$(<"$stderr_path")
  output=$(<"$stdout_path")
  if [[ "$rc" -eq 0 && -z "$error" && -n "$output" && "$output" != '[]' ]]; then
    inspect_state=present
  elif [[ "$rc" -eq 1 && ( -z "$output" || "$output" == '[]' ) ]] && \
    { [[ "$error" == "Error: No such object: $container" ]] || \
      [[ "$error" == "Error: No such container: $container" ]] || \
      [[ "$error" == "Error response from daemon: No such container: $container" ]]; }; then
    inspect_state=absent
  else
    inspect_state=error
  fi
  printf '%s\n' "$inspect_state" >"$state_path"
}

# shellcheck disable=SC2329  # Invoked through cleanup traps.
remove_container() {
  local stdout_path="$run_root/receipts/container-rm.stdout"
  local stderr_path="$run_root/receipts/container-rm.stderr"
  local rc_path="$run_root/receipts/container-rm.rc"
  local state_path="$run_root/receipts/container-rm.state"
  local rc error output
  if "$docker" rm -f "$container" >"$stdout_path" 2>"$stderr_path"; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" >"$rc_path"
  error=$(<"$stderr_path")
  output=$(<"$stdout_path")
  if [[ "$rc" -eq 0 && -z "$error" && "$output" == "$container" ]]; then
    remove_state=removed
  elif [[ -z "$output" ]] && \
    { [[ "$error" == "Error: No such object: $container" ]] || \
      [[ "$error" == "Error: No such container: $container" ]] || \
      [[ "$error" == "Error response from daemon: No such container: $container" ]]; } && \
    { [[ "$rc" -eq 0 ]] || [[ "$rc" -eq 1 ]]; }; then
    remove_state=already_absent
  else
    remove_state=error
  fi
  printf '%s\n' "$remove_state" >"$state_path"
}

actual_revision=$(git -C "$source_root" rev-parse HEAD)
source_status=$(git -C "$source_root" status --porcelain)
if [[ -n "$source_status" ]]; then
  printf 'W15 source authority is dirty: revision=%s status=%q\n' \
    "$actual_revision" "$source_status" >&2
  exit 2
fi
source_revision=$actual_revision

mapfile -t gpu_rows < <(nvidia-smi \
  --query-gpu=index,name,compute_cap,memory.used \
  --format=csv,noheader,nounits)
gpu_process_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l)
if [[ ${#gpu_rows[@]} -ne 8 || "$gpu_process_count" -ne 0 ]]; then
  printf 'GPU count/process preflight failed\n' >&2
  exit 2
fi
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r index name compute_cap memory_used <<<"$row"
  index=${index// /}
  name=${name# }
  compute_cap=${compute_cap// /}
  memory_used=${memory_used// /}
  if [[ "$index" -lt 0 || "$index" -gt 7 || \
    "$name" != 'NVIDIA RTX PRO 6000 Blackwell Server Edition' || \
    "$compute_cap" != 12.0 || "$memory_used" -ge 1024 ]]; then
    printf 'GPU inventory mismatch: %s\n' "$row" >&2
    exit 2
  fi
done
available_kib=$(df -Pk "$workspace" | awk 'NR==2 {print $4}')
available_bytes=$((available_kib * 1024))
if (( available_bytes < 85899345920 )); then
  printf 'workspace headroom below 80 GiB: %s bytes\n' "$available_bytes" >&2
  exit 2
fi

python3 - "$run_root/receipts/preflight.json" "$arm" "$phase" "$max_epochs" \
  "$available_bytes" "$source_revision" "$actual_config_sha" <<'PY'
import json, pathlib, sys
path, arm, phase, max_epochs, headroom, revision, config_sha = sys.argv[1:]
value = {
    "schema": "rlinf.w15.rtx6000-preflight/v1",
    "status": "passed",
    "profile": "absolute_correctness_b8",
    "arm": arm,
    "phase": phase,
    "max_epochs": int(max_epochs),
    "gpu_count": 8,
    "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "compute_capability": "12.0",
    "gpu_process_count": 0,
    "storage_headroom_bytes": int(headroom),
    "source_revision": revision,
    "config_sha256": config_sha,
}
pathlib.Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

asset_cache="$run_root/scratch/tmp/Assets"
mkdir -p "$asset_cache"
cp -a "$asset_seed/." "$asset_cache/"
chmod -R u+rwX "$asset_cache"
cp "$config" "$run_root/config.yaml"
cp "$0" "$run_root/launcher.sh"
sha256sum "$run_root/config.yaml" "$run_root/launcher.sh" "$contract" \
  "$extension" "$assets_override" \
  "$artifact_root/backbone-engines/rlinf-engine-receipt.json" \
  "$artifact_root/dit-engines/rlinf-refittable-dit-engine-receipt.json" \
  "$artifact_root/refittable-dit-parameter-map.json" \
  >"$run_root/receipts/inputs.sha256"

inspect_container prelaunch
if [[ "$inspect_state" != absent ]]; then
  printf 'cannot prove fresh container name is absent: %s\n' "$inspect_state" >&2
  exit 2
fi

sampler_pid=
# shellcheck disable=SC2329  # Invoked through cleanup traps.
cleanup() {
  rc=$?
  set +e
  touch "$run_root/gpu/stop"
  sampler_wait_rc=0
  if [[ -n "$sampler_pid" ]]; then
    wait "$sampler_pid"
    sampler_wait_rc=$?
  fi
  printf '%s\n' "$sampler_wait_rc" >"$run_root/gpu/sampler-wait.rc"
  python3 - "$run_root/gpu/samples.csv" "$run_root/receipts/gpu-sampler.json" <<'PY'
import csv, json, pathlib, sys
from collections import defaultdict
csv_path, receipt_path = map(pathlib.Path, sys.argv[1:])
expected = ["timestamp", "index", "memory_used_mib", "utilization_gpu_pct", "power_w"]
errors = []
groups = defaultdict(dict)
try:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            errors.append(f"unexpected header: {reader.fieldnames!r}")
        else:
            for line_number, row in enumerate(reader, 2):
                try:
                    timestamp = row["timestamp"]
                    index = int(row["index"])
                    if not timestamp or index not in range(8) or index in groups[timestamp]:
                        raise ValueError("invalid timestamp or GPU index")
                    groups[timestamp][index] = {
                        "memory_used_mib": float(row["memory_used_mib"]),
                        "utilization_gpu_pct": float(row["utilization_gpu_pct"]),
                        "power_w": float(row["power_w"]),
                    }
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"line {line_number}: {error}")
except OSError as error:
    errors.append(str(error))
complete = {ts: rows for ts, rows in groups.items() if set(rows) == set(range(8))}
if not complete:
    errors.append("no complete 8-GPU timestamp group")
peaks = {
    str(index): max(rows[index]["memory_used_mib"] for rows in complete.values())
    for index in range(8)
} if complete else {}
value = {
    "schema": "rlinf.w15.gpu-sampler/v1",
    "status": "passed" if not errors else "failed",
    "complete_timestamp_groups": len(complete),
    "observed_timestamp_groups": len(groups),
    "per_gpu_peak_memory_mib": peaks,
    "errors": errors,
}
receipt_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
raise SystemExit(0 if not errors else 1)
PY
  sampler_validation_rc=$?
  inspect_container pre-remove
  pre_remove_state=$inspect_state
  if [[ "$pre_remove_state" == present ]]; then
    cp "$run_root/receipts/container-pre-remove-inspect.stdout" \
      "$run_root/receipts/container-inspect.json"
    "$docker" logs "$container" >"$run_root/receipts/container.log" 2>&1
  fi
  remove_container
  inspect_container post-remove
  post_remove_state=$inspect_state
  removal_safe=0
  if [[ "$pre_remove_state" == present && "$remove_state" == removed && \
    "$post_remove_state" == absent ]]; then
    removal_safe=1
  elif [[ "$pre_remove_state" == absent && "$remove_state" == already_absent && \
    "$post_remove_state" == absent ]]; then
    removal_safe=1
  fi
  set -o pipefail
  nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    2>"$run_root/receipts/post-gpu-query.stderr" | sed '/^$/d' \
    | wc -l >"$run_root/receipts/post-gpu-query.stdout"
  post_gpu_query_rc=$?
  set +o pipefail
  post_gpu_process_count=$(<"$run_root/receipts/post-gpu-query.stdout")
  [[ "$post_gpu_process_count" =~ ^[0-9]+$ ]] || post_gpu_process_count=-1
  if [[ "$sampler_wait_rc" -ne 0 || "$sampler_validation_rc" -ne 0 || \
    "$removal_safe" -ne 1 || "$post_gpu_query_rc" -ne 0 || \
    "$post_gpu_process_count" -ne 0 ]]; then
    rc=1
  fi
  python3 - "$run_root/receipts/capacity.json" "$rc" "$sampler_wait_rc" \
    "$sampler_validation_rc" "$pre_remove_state" "$remove_state" \
    "$post_remove_state" "$post_gpu_query_rc" "$post_gpu_process_count" <<'PY'
import json, pathlib, sys
(path, exit_code, sampler_wait_rc, sampler_validation_rc, pre_remove_state,
 remove_state, post_remove_state, post_gpu_query_rc, post_gpu_process_count) = sys.argv[1:]
value = {
    "schema": "rlinf.w15.capacity/v1",
    "status": "passed" if int(exit_code) == 0 else "failed",
    "exit_code": int(exit_code),
    "sampler_wait_exit_code": int(sampler_wait_rc),
    "sampler_validation_exit_code": int(sampler_validation_rc),
    "container_pre_remove_state": pre_remove_state,
    "container_remove_state": remove_state,
    "container_post_remove_state": post_remove_state,
    "post_gpu_query_exit_code": int(post_gpu_query_rc),
    "post_gpu_process_count": int(post_gpu_process_count),
}
pathlib.Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
  writer_rc=$?
  [[ "$writer_rc" -eq 0 ]] || rc=1
  printf '%s\n' "$rc" >"$run_root/exit"
  trap - EXIT INT TERM
  exit "$rc"
}
trap cleanup EXIT INT TERM

printf 'timestamp,index,memory_used_mib,utilization_gpu_pct,power_w\n' \
  >"$run_root/gpu/samples.csv"
(
  while [[ ! -f "$run_root/gpu/stop" ]]; do
    timestamp=$(date -u +%FT%TZ)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits | sed "s/^/$timestamp,/"
    sleep 1
  done
) >>"$run_root/gpu/samples.csv" 2>"$run_root/gpu/sampler.err" &
sampler_pid=$!

set +e
"$docker" run --name "$container" \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  --network none \
  --entrypoint /bin/bash \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HOME=/w15-run/scratch/home \
  -e USER=liweim -e LOGNAME=liweim \
  -e TMPDIR=/w15-run/scratch/tmp \
  -e XDG_CACHE_HOME=/w15-run/scratch/cache \
  -e TORCHINDUCTOR_CACHE_DIR=/w15-run/scratch/torchinductor \
  -e RAY_TMPDIR=/w15-run/scratch/ray \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH=/w96-overlay:/w96-trt-runtime:/workspace/gr00t-n17:/workspace/rlinf-src \
  -e HF_HOME=/w15-run/scratch/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e ISAAC_PATH=/isaac-sim -e EXP_PATH=/isaac-sim/apps \
  -e CARB_APP_PATH=/isaac-sim/kit \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e NO_ALBUMENTATIONS_UPDATE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RAY_DEDUP_LOGS=0 -e RLINF_CODE_WORKING_DIR=0 \
  -e RLINF_EXT_MODULE=toolkits.gr00t_trocar.vulkan_extension \
  -e RLINF_CONFIG_FILE="$container_config" \
  -e W77_BACKBONE_MODEL_ROOT=/w96-model-inputs/Cosmos-Reason2-2B \
  -e W77_TROCAR_METADATA=/w96-inputs/trocar/metadata.json \
  -v "$source_root:/workspace/rlinf-src:ro" \
  -v "$gr00t_source:/workspace/gr00t-n17:ro" \
  -v "$python_overlay:/w96-overlay:ro" \
  -v "$trt_runtime:/w96-trt-runtime:ro" \
  -v "$graphics_runtime:/w12-driver:ro" \
  -v "$model_root/GR00T-N1.7-3B:/models/GR00T-N1.7-3B:ro" \
  -v "$model_root/Cosmos-Reason2-2B:/w96-model-inputs/Cosmos-Reason2-2B:ro" \
  -v "$run_root/config.yaml:$container_config:ro" \
  -v "$extension:$container_extension:ro" \
  -v "$assets_override:$container_assets:ro" \
  -v "$metadata:/w96-inputs/trocar/metadata.json:ro" \
  -v "$artifact_root:/w15-artifacts:ro" \
  -v "$asset_cache:/tmp/Assets:rw" \
  -v "$run_root/scratch/kit-cache:/isaac-sim/kit/cache:rw" \
  -v "$run_root/scratch/kit-data:/isaac-sim/kit/data:rw" \
  -v "$run_root/scratch/rlinf-entry-logs:/workspace/isaaclab/scripts/reinforcement_learning/rlinf/logs:rw" \
  -v "$run_root:/w15-run:rw" \
  -v "$run_root/output:/workspace/isaaclab/output:rw" \
  "$image" -lc "
set -euo pipefail
export LD_LIBRARY_PATH=\"/w12-driver:/w96-trt-runtime/tensorrt_libs:\${LD_LIBRARY_PATH:-}\"
cd /workspace/isaaclab
./isaaclab.sh train \\
  --rl_library rlinf \\
  --config_name isaaclab_ppo_gr00t_assemble_trocar_prod \\
  --model_path /models/GR00T-N1.7-3B \\
  --num_envs 64 \\
  --max_epochs $max_epochs
" >"$run_root/run.out" 2>"$run_root/run.err"
run_rc=$?
set -e
exit "$run_rc"
