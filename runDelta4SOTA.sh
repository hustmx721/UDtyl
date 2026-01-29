#!/usr/bin/env bash
set -euo pipefail

echo "Cross-session transfer experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("BrainprintNet" "MSNet" "FBCNet" "FBMSNet")
gpus=(1 2 3 4)

max_jobs=12
jobs=()
job_idx=0
failed=0

cleanup() {
  if ((${#jobs[@]} > 0)); then
    echo "Cleaning up ${#jobs[@]} running jobs..."
    for pid in "${jobs[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT INT TERM

wait_one() {
  local pid="$1"
  if ! wait "$pid"; then
    echo "WARNING: job failed (pid=$pid)" >&2
    failed=1
  fi
}

for dataset in "${datasets[@]}"; do
  for tgt_model in "${models[@]}"; do
    gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
    job_idx=$((job_idx + 1))

    echo "Launch: dataset=${dataset}, tgt_model=${tgt_model}, gpu=${gpu_id}"

    python -u main_UID.py \
      --dataset "${dataset}" \
      --gpuid "${gpu_id}" \
      --model "${tgt_model}"&

    pid=$!
    jobs+=("$pid")

    if (( ${#jobs[@]} >= max_jobs )); then
      wait_one "${jobs[0]}"
      jobs=("${jobs[@]:1}")
    fi
  done
done

for pid in "${jobs[@]}"; do
  wait_one "$pid"
done

if (( failed == 1 )); then
  echo "All experiments completed, BUT some jobs failed." >&2
  exit 1
fi

echo "All cross-session transfer experiments completed."
