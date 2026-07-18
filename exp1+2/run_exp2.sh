#!/usr/bin/env bash
# 这一行叫 shebang，用来告诉操作系统：这个脚本应当使用 bash 执行。
# /usr/bin/env 会从当前环境的 PATH 中寻找 bash，比直接写死 /bin/bash 更通用。


set -u
# set -u 表示：脚本使用未定义变量时立即报错并退出。
# 例如把 RESULTS_DIR 拼错成 RESULT_DIR 时，脚本不会把空字符串继续当作路径使用。

set -o pipefail
# run_exp2.sh 最后的训练命令使用了管道：python main.py ... | tee ...。
# Bash 默认只使用管道最后一个程序 tee 的退出状态；这样即使 Python 训练失败，
# 只要 tee 正常退出，整个脚本仍可能显示成功。
# pipefail 会让管道中任意程序失败时，整条管道返回失败状态。


##############################################################################
# 路径配置：保证从任意目录调用脚本时，都在 exp1+2 目录中执行实验 2       #
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BASH_SOURCE[0] 表示当前脚本文件自身的路径。
# dirname 取出脚本所在的文件夹；cd 进入该文件夹；pwd 得到绝对路径。
# $(...) 叫命令替换：先运行括号里的命令，再把输出结果赋给 SCRIPT_DIR。
#
# 例如脚本位于：
#   /home/user/intern1/exp1+2/run_exp2.sh
# 那么 SCRIPT_DIR 就是：
#   /home/user/intern1/exp1+2
#
# 这样即使在别的目录执行 /path/to/run_exp2.sh，脚本仍能找到 main.py。

VENV_PATH="${VENV_PATH:-$HOME/pyenv}"
# VENV_PATH 保存 Python 虚拟环境路径。
# ${变量:-默认值} 表示：
# - 如果外部已经设置 VENV_PATH，就使用外部传入的路径；
# - 如果没有设置，就默认使用 $HOME/pyenv，也就是 ~/pyenv。
#
# 如果虚拟环境不在默认位置，可以这样运行：
#   VENV_PATH=/path/to/venv ./run_exp2.sh

RESULTS_DIR="$SCRIPT_DIR/results_analysis_exp2"
# 实验 2 的结果目录。
# 它与实验 1 的 results_analysis_exp1 分开，防止两个实验的日志互相覆盖。

OUT_FILE="$RESULTS_DIR/run_exp2_out.txt"
# OUT_FILE 是本次训练的完整终端日志文件路径。
# 训练时显示的 Epoch、Loss、训练准确率和测试准确率都会写入这个文件。


cd "$SCRIPT_DIR"
# 进入脚本所在的 exp1+2 目录。
# main.py、data.py、models/ 和 weights/ 中使用了一些相对路径，
# 因此统一工作目录可以避免从其他位置启动脚本时找不到文件。

mkdir -p "$RESULTS_DIR"
# 创建实验 2 的结果目录。
# -p 表示父目录不存在时一起创建；目录已经存在时也不会报错。
# 必须先创建目录，后面的 tee 才能写入 run_exp2_out.txt。


##############################################################################
# 检查并激活 Python 虚拟环境                                           #
##############################################################################

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    # [ ... ] 是 shell 的条件测试语法。
    # -f 用来判断目标是不是一个普通文件；! 表示对判断结果取反。
    # 因此这里的含义是：如果虚拟环境的 activate 文件不存在，就进入报错分支。

    echo "Virtual environment not found: $VENV_PATH"
    # 显示当前尝试使用但没有找到的虚拟环境路径。

    echo "Set it with: VENV_PATH=/path/to/venv ./run_exp2.sh"
    # 告诉用户如何在运行脚本时指定其他虚拟环境。

    exit 1
    # exit 1 表示脚本异常结束。
    # 没有可用的 Python 环境时不继续训练，以免使用错误的 torch 版本。
fi

source "$VENV_PATH/bin/activate"
# source 在当前 shell 中执行 activate 文件，从而激活虚拟环境。
# 激活后，下面的 python 会优先使用该虚拟环境中的解释器、PyTorch 和 torchvision。


##############################################################################
# 强制检查 NVIDIA CUDA GPU                                              #
##############################################################################

python -c "import torch, sys; ok=torch.cuda.is_available(); print('CUDA GPU:', torch.cuda.get_device_name(0) if ok else 'unavailable'); sys.exit(0 if ok else 1)" || {
    # python -c "..." 表示直接执行引号中的一小段 Python 代码：
    #
    # 1. import torch, sys
    #    导入 PyTorch 和 Python 的 sys 模块。
    #
    # 2. ok = torch.cuda.is_available()
    #    检查当前 PyTorch 是否能访问 CUDA GPU。
    #
    # 3. torch.cuda.get_device_name(0)
    #    如果 CUDA 可用，打印编号为 0 的 GPU 名称，作为使用 GPU 的记录。
    #
    # 4. sys.exit(0 if ok else 1)
    #    CUDA 可用时返回状态 0；不可用时返回状态 1。
    #
    # || 的意思是“如果左边命令失败，就执行右边大括号中的命令”。
    # 因此这个检查确保实验不会在没有 CUDA 的情况下悄悄改用 CPU。

    echo "Experiment 2 requires an NVIDIA CUDA GPU."
    # 提示实验 2 必须使用支持 CUDA 的 NVIDIA GPU。

    exit 1
    # GPU 不可用时立即退出，不启动 200 个 epoch 的训练。
}


##############################################################################
# 组织实验 2 的训练命令                                                #
##############################################################################

CMD=(
    # CMD=(...) 定义一个 Bash 数组，每一行是命令或一个独立参数。
    # 使用数组可以正确处理空格和特殊字符，也便于给每个参数分别写注释。

    python main.py
    # 使用当前虚拟环境的 Python 执行训练入口 main.py。

    --model alexnet
    # 指定模型名称为 alexnet。
    # main.py 会把字符串 alexnet 交给 models/model_factory_dict.py，
    # 最终创建 models/alexnet.py 中定义的 AlexNet 对象。

    --steps 200
    # 总共训练 200 个 epoch。
    # main.py 中参数名沿用了 steps，但在本项目中一次 step 实际代表一个完整 epoch。

    --lr 0.1
    # 设置初始学习率为 0.1。
    # Trainer 会把它传给 SGD 优化器，并在训练过程中按学习率策略调整。

    --gpu
    # 通知 main.py 把模型和输入数据移动到 CUDA GPU。
    # 前面的 CUDA 检查已经保证此处不会在 GPU 不可用时继续训练。
)


##############################################################################
# 启动训练并保存完整日志                                                #
##############################################################################

"${CMD[@]}" 2>&1 | tee "$OUT_FILE"
# "${CMD[@]}" 会把 CMD 数组中的每一项作为独立参数传给 Python，等价于：
#   python main.py --model alexnet --steps 200 --lr 0.1 --gpu
#
# 2>&1 表示把标准错误 stderr 合并到标准输出 stdout。
# 因此正常训练信息和报错信息都会进入后面的 tee。
#
# | 是管道，把左边 Python 程序的输出传给右边的 tee。
# tee 同时完成两件事：
# - 继续在终端实时显示训练过程；
# - 把相同内容写入 results_analysis_exp2/run_exp2_out.txt。
#
# 因为脚本开头启用了 set -o pipefail，所以 Python 训练一旦失败，
# 即使 tee 成功写入日志，run_exp2.sh 最终仍会返回失败状态。

TRAIN_STATUS=${PIPESTATUS[0]}
# PIPESTATUS 是 Bash 保存管道中各个命令退出状态的数组。
# [0] 对应管道左侧的 Python 训练程序，[1] 对应 tee。
# 这里先保存训练程序的状态，避免后续分析命令覆盖它。


##############################################################################
# 从训练日志生成实验 2 的 CSV 指标表和训练曲线                         #
##############################################################################

echo "Analyzing experiment 2 training log..."

python "$RESULTS_DIR/analyze_exp2_log.py"
# analyze_exp2_log.py 会读取 run_exp2_out.txt，并生成：
# - exp2_epoch_metrics.csv：每个 epoch 的学习率、训练/测试 loss 和准确率；
# - exp2_accuracy_curve.png 或 .svg：训练与测试准确率曲线；
# - exp2_loss_curve.png 或 .svg：训练与测试 loss 曲线。

ANALYSIS_STATUS=$?
# 保存日志分析脚本的退出状态。

if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "Experiment 2 training failed with status $TRAIN_STATUS."
    exit "$TRAIN_STATUS"
fi
# 如果训练失败，返回训练程序原本的错误状态。

if [ "$ANALYSIS_STATUS" -ne 0 ]; then
    echo "Experiment 2 log analysis failed with status $ANALYSIS_STATUS."
    exit "$ANALYSIS_STATUS"
fi
# 训练成功但日志分析失败时，也让脚本明确返回失败状态。

echo "Experiment 2 metrics: $RESULTS_DIR/exp2_epoch_metrics.csv"
# 最后打印 CSV 的保存位置，方便训练结束后直接找到结果。
