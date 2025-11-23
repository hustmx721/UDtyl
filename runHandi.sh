echo "Handcrafted UD Experiments"

# Datasets, models, and handcrafted UD methods to evaluate
datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
handis=("sn" "stft" "rand")
handi_alpha=0.05

gpus=(0 1 2 3 4 5 6)
max_jobs=12
jobs=()

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for handi in "${handis[@]}"; do
      gpu_id=${gpus[$(( ( ${#dataset} + ${#model} + ${#handi} ) % ${#gpus[@]} ))]}

      # Task and UID classification with handcrafted UD templates
      python -u main_handi.py \
        --dataset="$dataset" \
        --gpuid="$gpu_id" \
        --model="$model" \
        --handi_method="$handi" \
        --handi_alpha="$handi_alpha" &
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
