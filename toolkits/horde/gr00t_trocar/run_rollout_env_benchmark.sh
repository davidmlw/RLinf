#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if (($# < 2)) || [[ $1 != --run-root ]]; then
  echo "usage: $0 --run-root PATH [rollout_env_benchmark.py arguments]" >&2
  exit 2
fi
run_root=$2
shift 2
[[ ! -e $run_root ]]

source_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source_sha=$(git -C "$source_root" rev-parse HEAD)
source_status=$(git -C "$source_root" status --short)
if [[ -n $source_status ]]; then
  echo "source checkout is dirty" >&2
  exit 2
fi

venv=${W04_NATIVE_VENV:?set W04_NATIVE_VENV}
asset_mirror=${W04_ASSET_MIRROR:?set W04_ASSET_MIRROR}
gr00t_source=${W04_GR00T_SOURCE:?set W04_GR00T_SOURCE}
model=${W04_MODEL:?set W04_MODEL}
[[ -x $venv/bin/python ]]
[[ -d $asset_mirror ]]
[[ -d $gr00t_source ]]
[[ -d $model ]]

# The benchmark is executed by file path, so Python would otherwise expose only
# the toolkit directory. Keep the selected RLinf checkout authoritative for
# dynamically loaded backend modules and their package imports.
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$run_root/tmp" "$run_root/cache"
exec > >(tee "$run_root/stdout.log") 2> >(tee "$run_root/stderr.log" >&2)
printf '%q ' "$0" --run-root "$run_root" "$@" >"$run_root/command.sh"
printf '\n' >>"$run_root/command.sh"
cat >"$run_root/launch.env" <<EOF
started_at=$(date -Iseconds)
source_sha=$source_sha
python=$venv/bin/python
asset_mirror=$(realpath "$asset_mirror")
gr00t_source=$(realpath "$gr00t_source")
model=$(realpath "$model")
pythonpath=$PYTHONPATH
EOF
sha256sum \
  "$source_root/toolkits/horde/gr00t_trocar/native-runtime-spec.json" \
  "$source_root/toolkits/horde/gr00t_trocar/rollout_env_benchmark.py" \
  >"$run_root/source-files.sha256"
"$venv/bin/python" -m pip freeze >"$run_root/pip-freeze.txt"

export TMPDIR="$run_root/tmp"
export XDG_CACHE_HOME="$run_root/cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONNOUSERSITE=1
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -lms 200 >"$run_root/gpu-samples.csv" &
sampler_pid=$!
printf '%s\n' "$sampler_pid" >"$run_root/gpu-sampler.pid"

set +e
timeout --signal=TERM --kill-after=60s 3600 \
  "$venv/bin/python" \
  "$source_root/toolkits/horde/gr00t_trocar/rollout_env_benchmark.py" \
  --rlinf-source "$source_root" \
  --gr00t-source "$gr00t_source" \
  --model "$model" \
  --asset-mirror "$asset_mirror" \
  --output "$run_root/receipt.json" \
  "$@"
python_rc=$?
kill -TERM "$sampler_pid" 2>/dev/null
wait "$sampler_pid"
sampler_rc=$?
set -e
printf '%s\n' "$python_rc" >"$run_root/python.exit"
printf '%s\n' "$sampler_rc" >"$run_root/gpu-sampler.exit"

sleep 1
set +e
pgrep -af "rollout_env_benchmark.py.*--output $run_root/receipt.json" \
  >"$run_root/post-owned-processes.txt"
process_rc=$?
set -e
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
  >"$run_root/post-gpu-processes.csv"

final_rc=0
if [[ $python_rc -ne 0 || ($sampler_rc -ne 0 && $sampler_rc -ne 143) || \
      $process_rc -eq 0 || ! -s $run_root/receipt.json ]]; then
  final_rc=1
fi
set +e
"$venv/bin/python" - "$run_root/receipt.json" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
if receipt.get("execution_status") != "passed":
    raise SystemExit(1)
if receipt.get("stage") not in {"cleanup_started", "completed"}:
    raise SystemExit(1)
PY
receipt_rc=$?
set -e
if [[ $receipt_rc -ne 0 ]]; then
  final_rc=1
fi

"$venv/bin/python" - \
  "$run_root" "$python_rc" "$sampler_rc" "$process_rc" "$receipt_rc" "$final_rc" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
used = []
with (root / "gpu-samples.csv").open(newline="") as stream:
    for row in csv.reader(stream):
        if len(row) >= 3:
            used.append(int(row[2].strip()))
summary = {
    "schema": "rlinf.w04.horde-rollout-env-attempt/v1",
    "status": "passed" if sys.argv[6] == "0" else "failed",
    "python_exit": int(sys.argv[2]),
    "gpu_sampler_exit": int(sys.argv[3]),
    "owned_process_check_exit": int(sys.argv[4]),
    "receipt_check_exit": int(sys.argv[5]),
    "sampled_gpu_peak_mib": max(used) if used else None,
    "receipt": json.loads((root / "receipt.json").read_text())
    if (root / "receipt.json").is_file()
    else None,
}
temporary = root / ".attempt.json.pending"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "attempt.json")
PY
printf '%s\n' "$final_rc" >"$run_root/exit"
exit "$final_rc"
