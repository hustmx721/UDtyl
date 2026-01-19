#!/usr/bin/env bash
set -euo pipefail

echo "Parameter Sensitivity Experiments (Pareto sweep)"

entry_script="${ENTRY_SCRIPT:-para_sensi.py}"

datasets=("Motor")
models=("ShallowConvNet")
gpus=(0)

seeds_per_sample="${SEEDS_PER_SAMPLE:-3}"
metric="${METRIC:-bca}"

epsilon_values=(0.1 0.5 1 5)
lambda_reg_values=(1e-4 1e-3 1e-2 1e-1)
lambda_uid_values=(1 2 4 8)
lambda_task_values=(0.5 1 2 4)

base_epsilon_delta="${BASE_EPSILON_DELTA:-1.0}"
base_lambda_reg="${BASE_LAMBDA_REG:-1e-3}"
base_lambda_uid="${BASE_LAMBDA_UID:-2.0}"
base_lambda_task="${BASE_LAMBDA_TASK:-1.0}"
p_eot="${P_EOT:-0.0}"

max_jobs="${MAX_JOBS:-1}"
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
    gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
    job_idx=$((job_idx + 1))

    echo "Launch: dataset=${dataset}, task_model=${model}, uid_model=${model}, gpu=${gpu_id}"

    python -u "${entry_script}" \
      --dataset "${dataset}" \
      --gpuid "${gpu_id}" \
      --task_model "${model}" \
      --uid_model "${model}" \
      --seeds-per-sample "${seeds_per_sample}" \
      --metric "${metric}" \
      --epsilon-values "${epsilon_values[@]}" \
      --lambda-reg-values "${lambda_reg_values[@]}" \
      --lambda-uid-values "${lambda_uid_values[@]}" \
      --lambda-task-values "${lambda_task_values[@]}" \
      --base-epsilon-delta "${base_epsilon_delta}" \
      --base-lambda-reg "${base_lambda_reg}" \
      --base-lambda-uid "${base_lambda_uid}" \
      --base-lambda-task "${base_lambda_task}" \
      --p-eot "${p_eot}" \
      --figure-path "figures/pareto_${dataset}_${model}.png" \
      --result-csv "csv/pareto_${dataset}_${model}.csv" &

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
  echo "All sensitivity runs completed, BUT some jobs failed." >&2
  exit 1
fi

echo "All parameter sensitivity experiments completed."
