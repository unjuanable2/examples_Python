#!/usr/bin/env bash

# 在 CIFAR-10 数据集上运行 ResNet18 实验：先运行 FP32，再运行 FP16。
# 两次实验的日志、CSV 指标和图片会分别保存在 results_analysis/ 目录中。

# 使用未定义变量时立即报错，避免变量名拼错后脚本仍继续运行。
set -u

# 管道中任意一个命令失败，都认为整个管道执行失败。
# 例如下面的“训练命令 | tee”中，即使 tee 成功，训练失败仍能被发现。
set -o pipefail

# BASH_SOURCE[0] 是当前脚本的位置。下面这行会取得脚本所在目录的绝对路径，
# 因而无论从哪个目录调用 run_exp.sh，都能正确找到 main.py 和结果目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 可在运行时通过 VENV_PATH 指定虚拟环境；未指定时默认使用 ~/pyenv。
VENV_PATH="${VENV_PATH:-$HOME/pyenv}"

# 所有实验输出统一放在脚本目录下的 results_analysis/ 中。
RESULTS_DIR="$SCRIPT_DIR/results_analysis"

# 切换到项目目录，并确保结果目录已经存在。
cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

# 检查 Python 虚拟环境是否存在。如果默认路径不对，可这样运行：
# VENV_PATH=/你的/虚拟环境路径 ./run_exp.sh
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Virtual environment not found: $VENV_PATH"
    echo "Set it with: VENV_PATH=/path/to/venv ./run_exp.sh"
    exit 1
fi

# 激活虚拟环境，让后续 python 命令使用其中安装的 Python 和依赖。
source "$VENV_PATH/bin/activate"

# 用 PyTorch 检查 CUDA 是否可用，并输出第 0 块 NVIDIA GPU 的名称。
# “|| { ... }”表示：如果前面的检查失败，就打印提示并退出脚本。
python -c "import torch, sys; ok=torch.cuda.is_available(); print('CUDA GPU:', torch.cuda.get_device_name(0) if ok else 'unavailable'); sys.exit(0 if ok else 1)" || {
    echo "The FP32/FP16 timing comparison requires an NVIDIA CUDA GPU."
    exit 1
}

# 执行一次指定精度的训练，并在训练结束后分析日志。
# 参数 precision 应为 fp32 或 fp16。
run_training() {
    # local 表示这些变量只在当前函数内有效。
    local precision="$1"
    local out_file="$RESULTS_DIR/run_exp1_out_${precision}.txt"

    # 用 Bash 数组保存命令及参数，可以避免字符串拼接引起的转义问题。
    # 两种精度实验共用这些基础参数：ResNet18、200 steps、学习率 0.2、GPU。
    local cmd=(
        python main.py
        --model resnet18
        --steps 200
        --lr 0.2
        --gpu
    )

    # FP16 实验额外启用半精度训练和 loss scaling；FP32 不添加这两个参数。
    if [ "$precision" = "fp16" ]; then
        cmd+=(--fp16 --loss_scaling)
    fi

    echo "============================================================"
    echo "Starting experiment 1 ${precision^^} training"
    echo "Log: $out_file"
    echo "============================================================"

    # 2>&1：把错误输出合并到标准输出。
    # tee：既在终端显示训练信息，也把同样的内容写入日志文件。
    "${cmd[@]}" 2>&1 | tee "$out_file"

    # PIPESTATUS[0] 保存管道中第一个命令（即训练命令）的退出状态。
    # 0 表示成功，非 0 表示训练失败。
    local train_status=${PIPESTATUS[0]}

    # 从刚生成的训练日志中提取指标，并生成 CSV 和相关图片。
    echo "Analyzing ${precision^^} training log..."
    python "$RESULTS_DIR/analyze_exp1_log.py" --precision "$precision"
    local analysis_status=$?

    # 分别检查训练和日志分析是否成功。任何一步失败都将返回对应错误码。
    if [ "$train_status" -ne 0 ]; then
        echo "${precision^^} training failed with status $train_status."
        return "$train_status"
    fi
    if [ "$analysis_status" -ne 0 ]; then
        echo "${precision^^} log analysis failed with status $analysis_status."
        return "$analysis_status"
    fi

    echo "Metrics: $RESULTS_DIR/exp1_epoch_metrics_${precision}.csv"
    echo "Accuracy figure: $RESULTS_DIR/exp1_accuracy_curve_${precision}.svg"
    echo "Loss figure: $RESULTS_DIR/exp1_loss_curve_${precision}.svg"
}

# 按顺序执行：FP32 完成后才开始 FP16。
# “|| exit $?”表示函数失败时，立即用相同的错误码退出整个脚本。
run_training fp32 || exit $?
run_training fp16 || exit $?

echo "FP32 and FP16 training completed successfully."
