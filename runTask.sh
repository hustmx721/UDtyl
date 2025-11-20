echo "Task Classify Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet" "1D_LSTM" "BrainprintNet" "MSNet")
gpus=(3 4 5 6)

max_jobs=12
jobs=()

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    gpu_id=${gpus[$(( ( ${#dataset} + ${#model} ) % ${#gpus[@]} ))]}
    
    # Start job in background
    python -u main.py --dataset="$dataset" --gpuid="$gpu_id" --model="$model" &
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