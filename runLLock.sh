echo "LLock Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
locktypes=("ires")
lock_epsilons=("1e-4" "5e-4" "1e-3" "5e-3")

gpus=(3 4 5 6)
max_jobs=3
jobs=()
job_idx=0

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for locktype in "${locktypes[@]}"; do
      for lock_epsilon in "${lock_epsilons[@]}"; do
        gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
        job_idx=$((job_idx + 1))

        echo "Launch: dataset=${dataset}, model=${model}, gpu=${gpu_id}, locktype=${locktype}, lock_epsilon=${lock_epsilon}"

        python -u main_LLock.py \
          --dataset="$dataset" \
          --gpuid="$gpu_id" \
          --model="$model" \
          --lock_type="$locktype" \
          --lock_epsilon="$lock_epsilon" \
          --no-save-models &
        jobs+=($!)

        if (( ${#jobs[@]} >= max_jobs )); then
          wait ${jobs[0]}
          jobs=("${jobs[@]:1}")
        fi
      done
    done
  done
done

wait

echo "All experiments completed."
