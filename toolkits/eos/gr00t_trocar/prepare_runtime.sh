#!/usr/bin/env bash
set -euo pipefail

required=(
  W73_SOURCE_ROOT
  W73_ISAACLAB_ROOT
  W73_GROOT_ROOT
  W73_RUNTIME_ROOT
  W73_RUNTIME_SPEC
  W73_RUNTIME_SPEC_SHA256
  W73_UV_CACHE
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'missing required environment variable: %s\n' "$name" >&2
    exit 2
  fi
done

test -d "$W73_SOURCE_ROOT/.git" || test -f "$W73_SOURCE_ROOT/.git"
test -d "$W73_ISAACLAB_ROOT/.git"
test -d "$W73_GROOT_ROOT/.git"
test -f "$W73_RUNTIME_SPEC"

actual_spec_sha=$(sha256sum "$W73_RUNTIME_SPEC" | awk '{print $1}')
if [[ "$actual_spec_sha" != "$W73_RUNTIME_SPEC_SHA256" ]]; then
  printf 'runtime spec SHA-256 mismatch: expected %s, found %s\n' \
    "$W73_RUNTIME_SPEC_SHA256" "$actual_spec_sha" >&2
  exit 2
fi

spec_value() {
  python3 - "$W73_RUNTIME_SPEC" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
print(value[sys.argv[2]])
PY
}

if [[ "$(spec_value schema)" != "rlinf.eos.python-runtime.v1" ]]; then
  printf 'unsupported runtime spec schema\n' >&2
  exit 2
fi
if [[ "$(spec_value isaaclab_revision)" != "$(git -C "$W73_ISAACLAB_ROOT" rev-parse HEAD)" ]]; then
  printf 'IsaacLab revision does not match runtime spec\n' >&2
  exit 2
fi
if [[ "$(spec_value gr00t_revision)" != "$(git -C "$W73_GROOT_ROOT" rev-parse HEAD)" ]]; then
  printf 'GR00T revision does not match runtime spec\n' >&2
  exit 2
fi
if [[ -n "$(git -C "$W73_SOURCE_ROOT" status --short)" ]]; then
  printf 'RLinf source must be clean before runtime preparation\n' >&2
  exit 2
fi

prepare_script_sha=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')
dependency_inputs_sha=$(
  git -C "$W73_SOURCE_ROOT" ls-files -s -- pyproject.toml requirements \
    | sha256sum | awk '{print $1}'
)

runtime_parent=$(dirname "$W73_RUNTIME_ROOT")
runtime_name=$(basename "$W73_RUNTIME_ROOT")
build_source="$runtime_parent/.${runtime_name}.source"
mkdir -p "$runtime_parent" "$W73_UV_CACHE"
exec 9>"$runtime_parent/.${runtime_name}.prepare.lock"
flock 9

manifest="$W73_RUNTIME_ROOT/rlinf-runtime-manifest.json"
if [[ -f "$manifest" ]]; then
  test -x "$W73_RUNTIME_ROOT/bin/python"
  test -f "$W73_RUNTIME_ROOT/requirements.freeze.txt"
  python3 - \
    "$manifest" \
    "$W73_RUNTIME_SPEC_SHA256" \
    "$W73_RUNTIME_ROOT/requirements.freeze.txt" \
    "$(git -C "$W73_ISAACLAB_ROOT" rev-parse HEAD)" \
    "$(git -C "$W73_GROOT_ROOT" rev-parse HEAD)" \
    "$prepare_script_sha" \
    "$dependency_inputs_sha" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("schema") != "rlinf.eos.python-runtime-manifest.v1":
    raise SystemExit("runtime manifest schema mismatch")
if manifest.get("runtime_spec_sha256") != sys.argv[2]:
    raise SystemExit("runtime manifest spec hash mismatch")
freeze = Path(sys.argv[3])
if hashlib.sha256(freeze.read_bytes()).hexdigest() != manifest.get(
    "requirements_freeze_sha256"
):
    raise SystemExit("runtime package freeze hash mismatch")
if manifest.get("isaaclab_revision") != sys.argv[4]:
    raise SystemExit("runtime IsaacLab revision mismatch")
if manifest.get("gr00t_revision") != sys.argv[5]:
    raise SystemExit("runtime GR00T revision mismatch")
if manifest.get("prepare_script_sha256") != sys.argv[6]:
    raise SystemExit("runtime prepare-script hash mismatch")
if manifest.get("rlinf_dependency_inputs_sha256") != sys.argv[7]:
    raise SystemExit("runtime RLinf dependency-input hash mismatch")
PY
  PYTHONPATH= "$W73_RUNTIME_ROOT/bin/python" - \
    "$(spec_value torch_version)" \
    "$(spec_value torchvision_version)" \
    "$(spec_value torchaudio_version)" \
    "$(spec_value torch_backend)" \
    "$(spec_value flash_attn_version)" <<'PY'
import sys

import flash_attn
import isaaclab_newton
import ray
import torch
import torchaudio
import torchvision

(
    expected_torch,
    expected_torchvision,
    expected_torchaudio,
    expected_backend,
    expected_flash_attn,
) = sys.argv[1:]
expected_build = f"{expected_torch}+{expected_backend}"
if not torch.__version__.startswith(expected_build):
    raise SystemExit(
        f"runtime Torch mismatch: expected {expected_build}, found {torch.__version__}"
    )
if flash_attn.__version__ != expected_flash_attn:
    raise SystemExit(
        "runtime FlashAttention mismatch: "
        f"expected {expected_flash_attn}, found {flash_attn.__version__}"
    )
if not torchvision.__version__.startswith(f"{expected_torchvision}+{expected_backend}"):
    raise SystemExit(
        "runtime torchvision mismatch: "
        f"expected {expected_torchvision}+{expected_backend}, "
        f"found {torchvision.__version__}"
    )
if not torchaudio.__version__.startswith(f"{expected_torchaudio}+{expected_backend}"):
    raise SystemExit(
        "runtime torchaudio mismatch: "
        f"expected {expected_torchaudio}+{expected_backend}, "
        f"found {torchaudio.__version__}"
    )
PY
  printf 'reusing verified RLinf runtime: %s\n' "$W73_RUNTIME_ROOT"
  exit 0
fi

if [[ -e "$W73_RUNTIME_ROOT" ]]; then
  rm -rf "$W73_RUNTIME_ROOT"
fi
umask 0002

cleanup_partial() {
  rc=$?
  set +e
  if [[ -e "$build_source" ]]; then
    git -C "$W73_SOURCE_ROOT" worktree remove --force "$build_source" \
      >/dev/null 2>&1 || rm -rf "$build_source"
  fi
  git -C "$W73_SOURCE_ROOT" worktree prune >/dev/null 2>&1
  if [[ $rc -ne 0 ]]; then
    rm -rf "$W73_RUNTIME_ROOT"
  fi
  trap - EXIT
  exit "$rc"
}
trap cleanup_partial EXIT

export UV_CACHE_DIR="$W73_UV_CACHE"
export UV_PYTHON_INSTALL_DIR="$runtime_parent/.uv-python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_TORCH_BACKEND="$(spec_value torch_backend)"
export ISAAC_LAB_PATH="$W73_ISAACLAB_ROOT"
export GR00T_PATH="$W73_GROOT_ROOT"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4
export NVCC_THREADS=4

python_version=$(spec_value python_version)
torch_version=$(spec_value torch_version)
if [[ -e "$build_source" ]]; then
  git -C "$W73_SOURCE_ROOT" worktree remove --force "$build_source" \
    >/dev/null 2>&1 || rm -rf "$build_source"
fi
git -C "$W73_SOURCE_ROOT" worktree prune
git -C "$W73_SOURCE_ROOT" worktree add \
  --detach "$build_source" "$(git -C "$W73_SOURCE_ROOT" rev-parse HEAD)"
cd "$build_source"
bash requirements/install.sh \
  --no-root \
  --platform nvidia \
  --python "$python_version" \
  --torch "$torch_version" \
  --venv "$W73_RUNTIME_ROOT" \
  --no-flash-attn \
  embodied --model gr00t --env isaaclab

git -C "$W73_SOURCE_ROOT" worktree remove --force "$build_source"
git -C "$W73_SOURCE_ROOT" worktree prune
if [[ -n "$(git -C "$W73_SOURCE_ROOT" status --short)" ]]; then
  printf 'runtime preparation changed the canonical RLinf source checkout\n' >&2
  git -C "$W73_SOURCE_ROOT" status --short >&2
  exit 2
fi

# IsaacLab pins its qualified Torch release during installation. Restore the
# runtime contract before compiling FlashAttention so its extension ABI is
# built exactly once against the final Torch version.
export PATH="$W73_RUNTIME_ROOT/bin:$PATH"
torch_index="https://download.pytorch.org/whl/$(spec_value torch_backend)"
uv pip install \
  --python "$W73_RUNTIME_ROOT/bin/python" \
  --index-url "$torch_index" \
  --upgrade \
  "torch==$(spec_value torch_version)" \
  "torchvision==$(spec_value torchvision_version)" \
  "torchaudio==$(spec_value torchaudio_version)"
uv pip uninstall --python "$W73_RUNTIME_ROOT/bin/python" flash-attn || true
UV_NO_CACHE=1 uv pip install \
  --python "$W73_RUNTIME_ROOT/bin/python" \
  "flash-attn==$(spec_value flash_attn_version)" \
  --no-build-isolation

uv pip freeze --python "$W73_RUNTIME_ROOT/bin/python" \
  >"$W73_RUNTIME_ROOT/requirements.freeze.txt"
freeze_sha=$(sha256sum "$W73_RUNTIME_ROOT/requirements.freeze.txt" | awk '{print $1}')

cd /
PYTHONPATH= "$W73_RUNTIME_ROOT/bin/python" - \
  "$manifest" "$W73_RUNTIME_SPEC_SHA256" "$freeze_sha" \
  "$(git -C "$W73_ISAACLAB_ROOT" rev-parse HEAD)" \
  "$(git -C "$W73_GROOT_ROOT" rev-parse HEAD)" \
  "$prepare_script_sha" \
  "$dependency_inputs_sha" \
  "$(spec_value torch_version)" \
  "$(spec_value torchvision_version)" \
  "$(spec_value torchaudio_version)" \
  "$(spec_value torch_backend)" \
  "$(spec_value flash_attn_version)" <<'PY'
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

import flash_attn
import isaaclab_newton
import ray
import torch
import torchaudio
import torchvision

manifest_path = Path(sys.argv[1])
spec_sha = sys.argv[2]
freeze_sha = sys.argv[3]
isaaclab_revision = sys.argv[4]
gr00t_revision = sys.argv[5]
prepare_script_sha = sys.argv[6]
dependency_inputs_sha = sys.argv[7]
expected_torch = sys.argv[8]
expected_torchvision = sys.argv[9]
expected_torchaudio = sys.argv[10]
expected_backend = sys.argv[11]
expected_flash_attn = sys.argv[12]

expected_build = f"{expected_torch}+{expected_backend}"
if not torch.__version__.startswith(expected_build):
    raise SystemExit(f"unexpected Torch build: {torch.__version__}")
if flash_attn.__version__ != expected_flash_attn:
    raise SystemExit(f"unexpected FlashAttention build: {flash_attn.__version__}")
if not torchvision.__version__.startswith(f"{expected_torchvision}+{expected_backend}"):
    raise SystemExit(f"unexpected torchvision build: {torchvision.__version__}")
if not torchaudio.__version__.startswith(f"{expected_torchaudio}+{expected_backend}"):
    raise SystemExit(f"unexpected torchaudio build: {torchaudio.__version__}")

value = {
    "schema": "rlinf.eos.python-runtime-manifest.v1",
    "runtime_spec_sha256": spec_sha,
    "prepare_script_sha256": prepare_script_sha,
    "rlinf_dependency_inputs_sha256": dependency_inputs_sha,
    "requirements_freeze_sha256": freeze_sha,
    "python": sys.version,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "torch_cuda": torch.version.cuda,
    "flash_attn": flash_attn.__version__,
    "ray": ray.__version__,
    "isaaclab": importlib.metadata.version("isaaclab"),
    "isaaclab_newton": importlib.metadata.version("isaaclab-newton"),
    "gr00t": importlib.metadata.version("gr00t"),
    "isaaclab_revision": isaaclab_revision,
    "gr00t_revision": gr00t_revision,
}
encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd, temporary_name = tempfile.mkstemp(prefix=".runtime-manifest.", dir=manifest_path.parent)
with os.fdopen(fd, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary_name, manifest_path)
PY

printf 'prepared RLinf runtime: %s\n' "$W73_RUNTIME_ROOT"
