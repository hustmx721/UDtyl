echo "EM Classify Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
gpus=(0 1 2 3 4 5 6)

max_jobs=12
jobs=()
job_idx=0

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    # gpu_id=${gpus[$(( ( ${#dataset} + ${#model} ) % ${#gpus[@]} ))]}
    gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
    job_idx=$((job_idx + 1))

    echo "Launch: dataset=${dataset}, model=${model}, hand_method=${handi}, gpu=${gpu_id}"

    # Start job in background
    python -u main_EM.py --dataset="$dataset" --gpuid="$gpu_id" --model="$model" &
    jobs+=($!) # Store the PID

    # Limit the number of running jobs
    if (( ${#jobs[@]} >= max_jobs )); then
        wait ${jobs[0]} # Wait for the first job to finish
        jobs=("${jobs[@]:1}") # Remove the first PID
    fi
  done
done

# Wait for any remaining jobs
wait

echo "All experiments completed."