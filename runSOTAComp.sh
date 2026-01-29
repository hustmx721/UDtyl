echo "SOTA Comparison Experiments"

datasets=("Motor")
models=("ShallowConvNet")

gpus=(6)
max_jobs=1
jobs=()
job_idx=0

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
    job_idx=$((job_idx + 1))

    echo "Launch: dataset=${dataset}, model=${model}, gpu=${gpu_id}"

    python -u main_SOTAComp.py \
      --dataset="$dataset" \
      --gpuid="$gpu_id" \
      --model="$model" \
      --no-save-models &
    jobs+=($!)

    if (( ${#jobs[@]} >= max_jobs )); then
        wait ${jobs[0]}
        jobs=("${jobs[@]:1}")
    fi
  done
done

wait

echo "All experiments completed."
