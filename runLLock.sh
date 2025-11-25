echo "LLock Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
locktypes=("linear" "ires")

gpus=(3 4 5 6)
max_jobs=2
jobs=()
job_idx=0 

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for locktype in "${locktypes[@]}"; do
      gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
      job_idx=$((job_idx + 1))

      echo "Launch: dataset=${dataset}, model=${model}, gpu=${gpu_id}, locktype=${locktype}"

      # Start job in background
      python -u main_LLock.py --dataset="$dataset" --gpuid="$gpu_id" --model="$model" --lock_type="$locktype" 
      jobs+=($!) # Store the PID

      # Limit the number of running jobs
      if (( ${#jobs[@]} >= max_jobs )); then
          wait ${jobs[0]} # Wait for the first job to finish
          jobs=("${jobs[@]:1}") # Remove the first PID
      fi
    done
  done
done

# Wait for any remaining jobs
wait

echo "All experiments completed."