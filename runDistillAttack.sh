#!/usr/bin/env bash
set -euo pipefail

# Batch runner for UID adversarial training on distilled UD data.
# It mirrors the EOT combinations used in distillation to ensure the
# correct STFT perturbation checkpoints are picked up by Distill_Attack.py.

echo "Distill Attack Experiments"

# Data / model grids
# datasets=("Rest" "Transient" "Steady" "Motor")
# models=("EEGNet" "DeepConvNet" "ShallowConvNet")
datasets=("Motor")
models=("ShallowConvNet")
# EOT disabled for all runs to match the request.
eot_flags=(
  # --disable_eot
  --eot_shift 0
  --eot_shift_prob 0.0
  --eot_channel_dropout 0.0 --eot_channel_dropout_prob 0.0
  --eot_scale_prob 0.0
  --eot_resample 0.05 --eot_resample_prob 1.0
)

gpus=(0 1 2 3 4 5)

# Multiple PGD settings. attack_alpha is left unset to use Distill_Attack.py's auto rule.
# Format: "eps steps norm random_start"
attack_configs=(
  "0.01 5 linf true"
  "0.03 5 linf true"
  "0.05 5 linf true"
  "0.01 5 l2 true"
  "0.03 5 l2 true"
  "0.05 5 l2 true"
)

max_jobs=6
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
    echo "Launch: dataset=${dataset}, model=${model}, attacks=${#attack_configs[@]}"

    for attack_cfg in "${attack_configs[@]}"; do
      read -r attack_eps attack_steps attack_norm attack_random <<<"${attack_cfg}"
      gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
      job_idx=$((job_idx + 1))

      python -u Distill_Attack.py \
        --dataset "${dataset}" \
        --gpuid "${gpu_id}" \
        --model "${model}" \
        --attack_eps "${attack_eps}" \
        --attack_steps "${attack_steps}" \
        --attack_norm "${attack_norm}" \
        "${attack_random:+--attack_random_start}" \
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
