#!/usr/bin/env bash
set -euo pipefail

echo "Distillation Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
# models=("EEGNet")
# Each entry enables exactly one EOT transform; others are disabled for that run.
eot_modes=(
  "none"
  "shift"
  "scale"
  "channel_dropout"
  "resample"
)
gpus=(3 4 5 6)

max_jobs=2
jobs=()
job_idx=0
failed=0

# Clean up background jobs on exit / ctrl+c
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
  for model in "${models[@]}"; do
    for eot in "${eot_modes[@]}"; do
      gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
      job_idx=$((job_idx + 1))

      # --------- Base: disable all transforms explicitly ----------
      case "$eot" in
        "none")
          eot_flags=(
            --eot_shift 0
            --eot_shift_prob 0.0
            --eot_scale_prob 0.0
            --eot_channel_dropout 0.0 --eot_channel_dropout_prob 0.0
            --eot_resample 0.0 --eot_resample_prob 0.0
          )
          ;;
        "shift")
          eot_flags=(
            # eot_shift 平移步长8~32; eot_shift_prob 平移概率 <=1.0
            --eot_shift 16
            --eot_shift_prob 1.0
            --eot_scale_min 0.0 --eot_scale_max 0.0 --eot_scale_prob 0.0
            --eot_channel_dropout 0.0 --eot_channel_dropout_prob 0.0
            --eot_resample 0.0 --eot_resample_prob 0.0
          )
          ;;
        "scale")
          eot_flags=(
            --eot_shift 0
            --eot_shift_prob 0.0
            --eot_scale
            --eot_scale_min 0.95 --eot_scale_max 1.05 --eot_scale_prob 1.0
            --eot_channel_dropout 0.0 --eot_channel_dropout_prob 0.0
            --eot_resample 0.0 --eot_resample_prob 0.0
          )
          ;;
        "channel_dropout")
          eot_flags=(
            --eot_shift 0
            --eot_shift_prob 0.0
            --eot_channel_dropout 0.1 --eot_channel_dropout_prob 1.0
            --eot_scale_prob 0.0
            --eot_resample 0.0 --eot_resample_prob 0.0
          )
          ;;
        "resample")
          eot_flags=(
            --eot_shift 0
            --eot_shift_prob 0.0
            --eot_channel_dropout 0.0 --eot_channel_dropout_prob 0.0
            --eot_scale_prob 0.0
            --eot_resample 0.05 --eot_resample_prob 1.0
          )
          ;;
        *)
          echo "Unknown EOT mode: $eot" >&2
          exit 1
          ;;
      esac

      echo "Launch: dataset=${dataset}, task_model=${model}, uid_model=${model}, eot=${eot}, gpu=${gpu_id}"

      python -u main_Distill.py \
        --dataset "${dataset}" \
        --gpuid "${gpu_id}" \
        --task_model "${model}" \
        --uid_model "${model}" \
        "${eot_flags[@]}" &

      pid=$!
      jobs+=("$pid")

      # Throttle
      if (( ${#jobs[@]} >= max_jobs )); then
        wait_one "${jobs[0]}"
        jobs=("${jobs[@]:1}")
      fi
    done
  done
done

# Wait remaining
for pid in "${jobs[@]}"; do
  wait_one "$pid"
done

if (( failed == 1 )); then
  echo "All experiments completed, BUT some jobs failed." >&2
  exit 1
fi

echo "All distillation experiments completed."
