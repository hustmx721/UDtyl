#!/usr/bin/env bash
set -euo pipefail

echo "Transfer Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
# eot_tags=("resample" "scale")
eot_tags=("scale")
gpus=(1 2 3)

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
  for src_model in "${models[@]}"; do
    for tgt_model in "${models[@]}"; do
      if [[ "$src_model" == "$tgt_model" ]]; then
        continue
      fi
      for eot in "${eot_tags[@]}"; do
        gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
        job_idx=$((job_idx + 1))

        eot_flags=(
          # resample
          # --eot_shift 0
          # --eot_shift_prob 0.0
          # --eot_channel_dropout 0.0 --eot_channel_dropout_prob 0.0
          # --eot_scale_prob 0.0
          # --eot_resample 0.05 --eot_resample_prob 1.0
          # scale
          --eot_shift 0
          --eot_shift_prob 0.0
          --eot_scale
          --eot_scale_min 0.95 --eot_scale_max 1.05 --eot_scale_prob 1.0
          --eot_channel_dropout 0.0 --eot_channel_dropout_prob 0.0
          --eot_resample 0.0 --eot_resample_prob 0.0
        )

        echo "Launch: dataset=${dataset}, src_model=${src_model}, tgt_model=${tgt_model}, eot=${eot}, gpu=${gpu_id}"

        python -u main_transfer.py \
          --dataset "${dataset}" \
          --gpuid "${gpu_id}" \
          --src_model "${src_model}" \
          --tgt_model "${tgt_model}" \
          "${eot_flags[@]}" &

        pid=$!
        jobs+=("$pid")

        if (( ${#jobs[@]} >= max_jobs )); then
          wait_one "${jobs[0]}"
          jobs=("${jobs[@]:1}")
        fi
      done
    done
  done
done

for pid in "${jobs[@]}"; do
  wait_one "$pid"
done

if (( failed == 1 )); then
  echo "All experiments completed, BUT some jobs failed." >&2
  exit 1
fi

echo "All transfer experiments completed."
