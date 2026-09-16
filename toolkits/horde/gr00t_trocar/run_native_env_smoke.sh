#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_native_env_smoke.sh --run-root PATH --num-envs N --steps N \
  --asset-mirror PATH --venv PATH [--timeout-seconds N]
EOF
}

run_root=
num_envs=
steps=
asset_mirror=
venv=
timeout_seconds=900
while (($#)); do
  case "$1" in
    --run-root) run_root=$2; shift 2 ;;
    --num-envs) num_envs=$2; shift 2 ;;
    --steps) steps=$2; shift 2 ;;
    --asset-mirror) asset_mirror=$2; shift 2 ;;
    --venv) venv=$2; shift 2 ;;
    --timeout-seconds) timeout_seconds=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in run_root num_envs steps asset_mirror venv; do
  if [[ -z ${!value} ]]; then
    echo "missing --${value//_/-}" >&2
    exit 2
  fi
done
[[ $num_envs =~ ^[1-9][0-9]*$ ]]
[[ $steps =~ ^[1-9][0-9]*$ ]]
[[ $timeout_seconds =~ ^[1-9][0-9]*$ ]]
[[ -x $venv/bin/python ]]
[[ -d $asset_mirror ]]
[[ ! -e $run_root ]]

mkdir -p "$run_root/tmp" "$run_root/cache"
exec > >(tee "$run_root/stdout.log") 2> >(tee "$run_root/stderr.log" >&2)

source_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source_sha=$(git -C "$source_root" rev-parse HEAD)
source_status=$(git -C "$source_root" status --short)
if [[ -n $source_status ]]; then
  echo "source checkout is dirty" >&2
  exit 2
fi

cat >"$run_root/launch.env" <<EOF
started_at=$(date -Iseconds)
source_sha=$source_sha
python=$venv/bin/python
num_envs=$num_envs
steps=$steps
asset_mirror=$(realpath "$asset_mirror")
timeout_seconds=$timeout_seconds
EOF
sha256sum \
  "$source_root/toolkits/horde/gr00t_trocar/native-runtime-spec.json" \
  "$source_root/toolkits/horde/gr00t_trocar/native_env_smoke.py" \
  >"$run_root/source-files.sha256"
"$venv/bin/python" -m pip freeze >"$run_root/pip-freeze.txt"

export TMPDIR="$run_root/tmp"
export XDG_CACHE_HOME="$run_root/cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONNOUSERSITE=1
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

set +e
timeout --signal=TERM --kill-after=30s "$timeout_seconds" \
  "$venv/bin/python" \
  "$source_root/toolkits/horde/gr00t_trocar/native_env_smoke.py" \
  --num-envs "$num_envs" \
  --steps "$steps" \
  --asset-mirror "$asset_mirror" \
  --output "$run_root/receipt.json"
python_rc=$?
set -e
printf '%s\n' "$python_rc" >"$run_root/python.exit"

sleep 1
set +e
pgrep -af "native_env_smoke.py.*--output $run_root/receipt.json" \
  >"$run_root/post-owned-processes.txt"
process_rc=$?
set -e
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
  >"$run_root/post-gpu-processes.csv"

final_rc=0
if [[ $python_rc -ne 0 || $process_rc -eq 0 || ! -s $run_root/receipt.json ]]; then
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

"$venv/bin/python" - "$run_root" "$python_rc" "$process_rc" "$receipt_rc" "$final_rc" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = {
    "schema": "rlinf.w04.horde-native-env-attempt/v1",
    "status": "passed" if sys.argv[5] == "0" else "failed",
    "python_exit": int(sys.argv[2]),
    "owned_process_check_exit": int(sys.argv[3]),
    "receipt_check_exit": int(sys.argv[4]),
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
