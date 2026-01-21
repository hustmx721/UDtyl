#!/usr/bin/env bash
set -euo pipefail

entry_script="${ENTRY_SCRIPT:-para_sensi.py}"

dataset="Motor"
model="ShallowConvNet"
gpus=(1 2 3 4 5 6)
max_jobs=6

seeds_per_sample=3
metric=bca
p_eot=1.0

# sweep values
epsilon_values=(0.1 0.5 1 5)
lambda_reg_values=(1e-4 1e-3 1e-2 1e-1)
lambda_uid_values=(1 2 5 8)
lambda_task_values=(0.5 1 2 4)

# base values
base_epsilon_delta=1.0
base_lambda_reg=1e-3
base_lambda_uid=5.0
base_lambda_task=1.0

mkdir -p figures csv logs

jobs=()
job_idx=0
failed=0

wait_one() {
  local pid="$1"
  if ! wait "$pid"; then failed=1; fi
}

run_job() {
  local tag="$1"
  local eps="$2"
  local lreg="$3"
  local luid="$4"
  local ltask="$5"

  local gpu=${gpus[$(( job_idx % ${#gpus[@]} ))]}
  job_idx=$((job_idx + 1))

  echo "Launch ${tag} on GPU ${gpu}"

  python -u "${entry_script}" \
    --dataset "${dataset}" \
    --gpuid "${gpu}" \
    --task_model "${model}" \
    --uid_model "${model}" \
    --seeds-per-sample "${seeds_per_sample}" \
    --metric "${metric}" \
    --epsilon-values "${eps}" \
    --lambda-reg-values "${lreg}" \
    --lambda-uid-values "${luid}" \
    --lambda-task-values "${ltask}" \
    --base-epsilon-delta "${base_epsilon_delta}" \
    --base-lambda-reg "${base_lambda_reg}" \
    --base-lambda-uid "${base_lambda_uid}" \
    --base-lambda-task "${base_lambda_task}" \
    --p-eot "${p_eot}" \
    --figure-path "figures/${tag}.png" \
    --result-csv "csv/${tag}.csv" \
    > "logs/${tag}.log" 2>&1 &

  jobs+=("$!")
  if (( ${#jobs[@]} >= max_jobs )); then
    wait_one "${jobs[0]}"
    jobs=("${jobs[@]:1}")
  fi
}

# 1) sweep epsilon
for eps in "${epsilon_values[@]}"; do
  run_job "sensi_eps_${eps}" "$eps" "$base_lambda_reg" "$base_lambda_uid" "$base_lambda_task"
done

# 2) sweep lambda_reg
for lreg in "${lambda_reg_values[@]}"; do
  run_job "sensi_lreg_${lreg}" "$base_epsilon_delta" "$lreg" "$base_lambda_uid" "$base_lambda_task"
done

# 3) sweep lambda_uid
for luid in "${lambda_uid_values[@]}"; do
  run_job "sensi_luid_${luid}" "$base_epsilon_delta" "$base_lambda_reg" "$luid" "$base_lambda_task"
done

# 4) sweep lambda_task
for ltask in "${lambda_task_values[@]}"; do
  run_job "sensi_ltask_${ltask}" "$base_epsilon_delta" "$base_lambda_reg" "$base_lambda_uid" "$ltask"
done

# wait remaining
for pid in "${jobs[@]}"; do
  wait_one "$pid"
done

if (( failed == 1 )); then
  echo "Some jobs failed." >&2
  exit 1
fi

echo "All done."
