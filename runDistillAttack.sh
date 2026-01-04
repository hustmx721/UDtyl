#!/usr/bin/env bash
set -euo pipefail

# Batch runner for UID adversarial training on distilled UD data.
# It mirrors the EOT combinations used in distillation to ensure the
# correct STFT perturbation checkpoints are picked up by Distill_Attack.py.

echo "Distill Attack Experiments"

# Data / model grids
datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")

# Each entry enables exactly one EOT transform; others are disabled for that run.
eot_modes=(
  "none"
  "shift"
  "scale"
  "channel_dropout"
  "resample"
)

gpus=(0 1 2)

# PGD defaults (aligned with Distill_Attack.py recommendations)
attack_eps=0.1
attack_steps=10
attack_norm="linf"
# We keep attack_alpha unset to allow the script default (1.5*eps/steps).

# Distillation hyperparameters used for locating the STFT delta.
# These must match the settings used when running main_Distill.py.
lambda_task=1.0
lambda_uid=5.0
lambda_reg=0.001

# Only run the base seed per job to match the saved delta filename.
seed=2024
repeats=1

max_jobs=3
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
  for model in "${models[@]}"; do
    for eot in "${eot_modes[@]}"; do
      gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
      job_idx=$((job_idx + 1))

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

      echo "Launch: dataset=${dataset}, model=${model}, eot=${eot}, gpu=${gpu_id}"

      python -u Distill_Attack.py \
        --dataset "${dataset}" \
        --gpuid "${gpu_id}" \
        --model "${model}" \
        --lambda_task "${lambda_task}" \
        --lambda_uid "${lambda_uid}" \
        --lambda_reg "${lambda_reg}" \
        --attack_eps "${attack_eps}" \
        --attack_steps "${attack_steps}" \
        --attack_norm "${attack_norm}" \
        --seed "${seed}" \
        --repeats "${repeats}" \
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

for pid in "${jobs[@]}"; do
  wait_one "$pid"
done

if (( failed == 1 )); then
  echo "All experiments completed, BUT some jobs failed." >&2
  exit 1
fi

echo "All distill attack experiments completed."
