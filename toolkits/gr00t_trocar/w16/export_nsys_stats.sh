#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s <nsys-bin> <run-root>\n' "$0" >&2
  exit 2
fi

nsys_bin=$1
run_root=$2
report_root="$run_root/nsights"
output_root="$run_root/analysis/nsys-stats"

[[ -x "$nsys_bin" ]] || { printf 'nsys is not executable: %s\n' "$nsys_bin" >&2; exit 2; }
[[ -d "$report_root" ]] || { printf 'missing report root: %s\n' "$report_root" >&2; exit 2; }
[[ ! -e "$output_root" ]] || { printf 'output root already exists: %s\n' "$output_root" >&2; exit 2; }

mapfile -t reports < <(find "$report_root" -maxdepth 1 -type f -name '*.nsys-rep' -print | sort)
if [[ ${#reports[@]} -ne 3 ]]; then
  printf 'expected exactly three reports, got %d\n' "${#reports[@]}" >&2
  exit 2
fi

mkdir -p "$output_root"
reports_csv='nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum,vulkan_api_sum'
for report in "${reports[@]}"; do
  name=$(basename "$report" .nsys-rep)
  mkdir -p "$output_root/$name"
  "$nsys_bin" stats \
    --force-overwrite true \
    --report "$reports_csv" \
    --format csv \
    --output "$output_root/$name/stats" \
    "$report" \
    >"$output_root/$name/stdout.log" \
    2>"$output_root/$name/stderr.log"
  rm -f "${report%.nsys-rep}.sqlite"
done

find "$report_root" "$output_root" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"$run_root/analysis/SHA256SUMS"
