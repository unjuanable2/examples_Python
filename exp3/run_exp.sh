#!/usr/bin/env bash
# 上面这一行叫作 shebang：执行 ./run_exp.sh 时，系统会通过环境变量 PATH 找到 Bash，
# 并使用 Bash 解释这个脚本。

# 一键运行 exp3 的 PyTorch FP32 baseline 与 TensorRT INT8 对比实验。
# 脚本会把 stdout 和 stderr 同时显示在终端，并完整保存到
# results_analysis/run_exp3_out.txt。

set -euo pipefail
# -e：任意命令执行失败时立即停止脚本；
# -u：使用未定义的变量时立即报错；
# -o pipefail：管道中任何一个命令失败，都把整条管道视为执行失败。
# 这三个选项可以防止实验在中途出错后仍继续运行并生成不完整的结果。

# 使用脚本自己的绝对路径，不依赖用户从哪个目录执行该脚本。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results_analysis"
LOG_FILE="$RESULTS_DIR/run_exp3_out.txt"

mkdir -p "$RESULTS_DIR"

# 从这一行开始，脚本本身、Python、PyTorch、CUDA 和 TensorRT 的 stdout/stderr
# 都会经过 tee：既保留终端显示，也覆盖写入本次实验日志。
exec > >(tee "$LOG_FILE") 2>&1

echo "============================================================"
echo "Exp3: AlexNet PyTorch FP32 vs TensorRT INT8"
echo "============================================================"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Script directory: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo "Warm-up runs: 50"
echo "Measured runs: 100"

# 先进入 exp3 目录再运行程序，避免 TensorRT 或其它第三方库使用相对路径时
# 找不到文件。int8_infer.py 内部固定让 FP32 和 INT8 各预热 50 次、
# 正式测量 100 次，并继续计算完整 CIFAR-10 测试集 accuracy。
cd "$SCRIPT_DIR"
PYTHONUNBUFFERED=1 python3 int8_infer.py \
    --gpu \
    --model alexnet

echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Experiment completed successfully."
