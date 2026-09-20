#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: run_w08.sh {te|tr} ATTEMPT" >&2
    exit 2
fi

ARM=$1
ATTEMPT=$2
case "$ARM" in
    te)
        CONFIG_NAME=isaaclab_franka_stack_cube_ppo_gr00t_w08_te
        MAX_STEPS=1
        ;;
    tr)
        CONFIG_NAME=isaaclab_franka_stack_cube_ppo_gr00t_w08_tr
        # Revision 1 is refitted and adopted only at the next weight sync.
        MAX_STEPS=2
        ;;
    *)
        echo "ARM must be te or tr" >&2
        exit 2
        ;;
esac

SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
REMOTE_WORKSPACE=${W08_REMOTE_WORKSPACE:-/home/liweim/rl-workspace/rlinf}
W06_ROOT=${W08_W06_ROOT:-$REMOTE_WORKSPACE/w06-n15-l20-trt-backbone}
TRT_ROOT=${W08_TRT_ROOT:-$REMOTE_WORKSPACE/runtime/W06/tensorrt-10.15.1.29}
BACKBONE_ROOT=$W06_ROOT/results/W06/W06-18-sm89-engines
DIT_ONNX_ROOT=$W06_ROOT/results/W06/W06-29-n15-refittable-dit-onnx
DIT_ENGINE_ROOT=$W06_ROOT/results/W06/W06-30-n15-refittable-dit-engine

export W08_BACKBONE_ENGINE_DIR=$BACKBONE_ROOT
export W08_BACKBONE_RECEIPT=$BACKBONE_ROOT/rlinf-engine-receipt.json
export W08_BACKBONE_RECEIPT_SHA256=c4abe3ad9a3da7dd03e1266609d89d426cfb4814671295e7432ea32311991c81
export W08_DIT_ENGINE=$DIT_ENGINE_ROOT/dit_bf16_refit.engine
export W08_DIT_RECEIPT=$DIT_ENGINE_ROOT/rlinf-refittable-dit-engine-receipt.json
export W08_DIT_RECEIPT_SHA256=80eb59d87e8ab30c97bf91762c9762ed121e32ac0d292dc46090cff7d3ea30ef
export W08_DIT_PARAMETER_MAP=$DIT_ONNX_ROOT/refittable-dit-parameter-map.json
export W08_DIT_PARAMETER_MAP_SHA256=de4e723b3134ceda2737c94be17b1640624c9ac38e35c116907e04251b4abd70
export W08_DIT_SOURCE_DIGEST_REVISION_0=cfbf9ccfca11bfcf56ed6f9d88e7823fe7a6f823a5218faf4b255db5aefaf8e5

for path in \
    "$TRT_ROOT/tensorrt" \
    "$TRT_ROOT/tensorrt_libs" \
    "$W08_BACKBONE_RECEIPT" \
    "$W08_DIT_ENGINE" \
    "$W08_DIT_RECEIPT" \
    "$W08_DIT_PARAMETER_MAP"; do
    if [[ ! -e "$path" ]]; then
        echo "required W08 input is missing: $path" >&2
        exit 1
    fi
done

export RUN_SERIES=W08
export CONFIG_NAME
export EXTRA_PYTHONPATH=$TRT_ROOT
export EXTRA_LD_LIBRARY_PATH=$TRT_ROOT/tensorrt_libs
export W05_TMP_ROOT=${W08_TMP_ROOT:-$HOME/r/w08}

exec "$SOURCE/toolkits/l20/gr00t_stack_cube/run_w05.sh" "$ATTEMPT" "$MAX_STEPS"
