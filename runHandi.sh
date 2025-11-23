echo "Handi UD Classify Experiments"

datasets=("Rest" "Transient" "Steady" "Motor")
models=("EEGNet" "DeepConvNet" "ShallowConvNet")
handis=("sn" "stft" "rand")
gpus=(0 1 2 3 4 5 6)

max_jobs=12
jobs=()

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for handi in "${handis[@]}"; do

      # 根据字符串长度做一个简单 hash 分配 GPU
      gpu_id=${gpus[$(( ( ${#dataset} + ${#model} + ${#handi} ) % ${#gpus[@]} ))]}

      # 启动后台任务：注意 hand_method 参数
      python -u main_EM.py \
        --dataset "$dataset" \
        --gpuid "$gpu_id" \
        --model "$model" \
        --handi_method "$handi" &

      # 记录 PID
      jobs+=($!)

      # 控制最大并发数
      if (( ${#jobs[@]} >= max_jobs )); then
        wait "${jobs[0]}"              # 等第一个任务结束
        jobs=("${jobs[@]:1}")          # 从数组中移除这个 PID
      fi

    done
  done
done

# 等待所有剩余任务结束
wait

echo "All experiments completed."
