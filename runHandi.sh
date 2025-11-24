echo "Handcrafted UD Experiments"

# Datasets, models, and handcrafted UD methods to evaluate
datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
handis=("sn" "stft" "rand")
handi_alpha=0.05

# Use small CPU thread counts to reduce contention when running many jobs in parallel.
# Set to 0 or leave empty to let PyTorch decide.
torch_threads=1
torch_interop_threads=1

gpus=(0 1 2 3 4 5 6)
max_jobs=12
jobs=()
job_idx=0                # 用来轮询分配 GPU

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for handi in "${handis[@]}"; do
      # gpu_id=${gpus[$(( ( ${#dataset} + ${#model} + ${#handi} ) % ${#gpus[@]} ))]}
      gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
      job_idx=$((job_idx + 1))

      echo "Launch: dataset=${dataset}, model=${model}, hand_method=${handi}, gpu=${gpu_id}"

      # Task and UID classification with handcrafted UD templates
      python -u main_Handi.py \
        --dataset="$dataset" \
        --gpuid="$gpu_id" \
        --model="$model" \
        --handi_method="$handi" \
        --handi_alpha="$handi_alpha" \
        --torch_threads="${torch_threads:-0}" \
        --torch_interop_threads="${torch_interop_threads:-0}" &
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
