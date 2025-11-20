# 模糊熵方法计算过于复杂, 效果一般, 不做实验
echo "Task Classify Experiments -- Machine Learning"

datasets=("Rest" "Transient" "Steady" "Motor")
fea_types=("wavelet" "PSD" "AR_burg" "MFCC")
clf_types=("SVM" "LDA")

max_jobs=20
jobs=()

# 启动第二部分的循环实验
for dataset in "${datasets[@]}"; do
  for fea_type in "${fea_types[@]}"; do
    for clf_type in "${clf_types[@]}"; do
      
      # 启动后台作业
      echo "启动实验: dataset=$dataset, fea_type=$fea_type, clf_type=$clf_type"
      python -u fea_main.py --dataset="$dataset" --fea_type="$fea_type" --clf_type="$clf_type" &
      jobs+=($!) # 存储PID
      
      # 限制同时运行的作业数量
      if (( ${#jobs[@]} >= max_jobs )); then
          echo "达到最大作业数，等待作业完成..."
          wait ${jobs[0]} # 等待第一个作业完成
          jobs=("${jobs[@]:1}") # 移除第一个PID
      fi
    done
  done
done

# 等待所有剩余作业完成
echo "所有作业已启动，等待完成..."
wait

echo "所有实验完成。"