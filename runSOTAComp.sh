echo "SOTA Comparison Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")

gpus=(0 1 2 3 4 5 6)
max_jobs=6
jobs=()
job_idx=0

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
    job_idx=$((job_idx + 1))

    echo "Launch: dataset=${dataset}, model=${model}, gpu=${gpu_id}"

    python -u main_SOTAComp.py --dataset="$dataset" --gpuid="$gpu_id" --model="$model" &
    jobs+=($!)

    if (( ${#jobs[@]} >= max_jobs )); then
        wait ${jobs[0]}
        jobs=("${jobs[@]:1}")
    fi
  done
done

wait

echo "All experiments completed."
