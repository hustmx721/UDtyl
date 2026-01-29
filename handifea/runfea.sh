# 模糊熵方法计算过于复杂, 效果一般, 不做实验
echo "Handicraft Features UID STFTPerturbation cross-session Classify Experiments -- Machine Learning"

datasets=("Rest" "Transient" "Steady" "Motor")
# fea_types=("wavelet" "PSD" "AR_burg" "MFCC")
# clf_types=("SVM" "LDA")

max_jobs=4
jobs=()

# 启动第二部分的循环实验
for dataset in "${datasets[@]}"; do
  # 启动后台作业
  echo "启动实验: dataset=$dataset"
  python -u fea_cross_session.py --dataset="$dataset" &
  jobs+=($!) # 存储PID
  
  # 限制同时运行的作业数量
  if (( ${#jobs[@]} >= max_jobs )); then
      echo "达到最大作业数，等待作业完成..."
      wait ${jobs[0]} # 等待第一个作业完成
      jobs=("${jobs[@]:1}") # 移除第一个PID
  fi
done

# 等待所有剩余作业完成
echo "所有作业已启动，等待完成..."
wait

echo "所有实验完成。"