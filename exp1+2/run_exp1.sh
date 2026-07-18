#!/usr/bin/env bash
# 这一行叫 shebang，告诉系统：这个脚本要用 bash 来执行。
# /usr/bin/env bash 会在当前环境里寻找 bash，比写死 /bin/bash 更灵活。

set -u
# 表示：如果脚本里使用了一个没有定义过的变量，就立刻报错退出。
# 这样可以避免因为变量名拼错导致脚本继续乱跑。

##############################################################################
# 这个脚本用于从 exp1 目录启动训练，并把终端输出保存到 results_analysis_exp1/ 文件夹 #
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 一个变量，保存这个脚本所在目录的绝对路径。
#
# 这一行可以拆开理解：
# - BASH_SOURCE[0]：当前脚本文件本身的路径
# - dirname "${BASH_SOURCE[0]}"：取出脚本所在的文件夹
# - cd "...": 进入这个文件夹
# - pwd：打印当前文件夹的绝对路径
# - $(...)：执行括号里的命令，并把输出结果放到这里
#
# 为什么要这样写？
# 因为你可能从任何目录运行这个脚本。
# 有了 SCRIPT_DIR，脚本总能先找到 exp1 目录，再从 exp1 里运行 main.py。

VENV_PATH="${VENV_PATH:-$HOME/pyenv}"
# VENV_PATH 是虚拟环境路径。
# - 如果运行脚本前已经设置了 VENV_PATH，就使用用户设置的值；
# - 如果没有设置，就默认使用 $HOME/pyenv。
#   - $HOME 是当前用户的 home 目录，所以默认虚拟环境路径就是 ~/pyenv。
#
# 如果你的虚拟环境不叫 ~/pyenv，可以这样临时指定：
#   VENV_PATH=/home/jianing/torch-env ./run_exp1.sh

RESULTS_DIR="$SCRIPT_DIR/results_analysis_exp1"
# RESULTS_DIR 是结果文件夹路径。
# 训练日志、CSV 表格、曲线图都会集中保存到这个文件夹里。

OUT_FILE="$RESULTS_DIR/run_exp1_out.txt"
# OUT_FILE 是输出日志文件路径。
# 这里表示把日志保存到 exp1/results_analysis_exp1/run_exp1_out.txt。

cd "$SCRIPT_DIR"
# cd 到脚本所在目录，也就是 exp1。
# 这样后面运行 python main.py 时，不管你从哪里启动脚本，
# Python 都会在 exp1 目录下运行。
#
# 双引号 "$SCRIPT_DIR" 是为了防止路径里有空格时出错。

mkdir -p "$RESULTS_DIR"
# mkdir 是创建文件夹的命令。
# -p 表示：
# - 如果 results_analysis_exp1 不存在，就创建；
# - 如果已经存在，也不要报错。
#
# 这一步要放在 tee 写日志之前，
# 因为 tee 不能往一个不存在的文件夹里写文件。

if [ ! -f "$VENV_PATH/bin/activate" ]; then
# if ... then ... fi 是 shell 的条件判断。
#
# [ ! -f "$VENV_PATH/bin/activate" ] 的意思是：
#   如果 "$VENV_PATH/bin/activate" 这个文件不存在，就进入下面的报错分支
# - -f 用来判断某个路径是不是一个普通文件；
# - ! 表示取反
#
# Python 虚拟环境通常会有一个 activate 文件：~/pyenv/bin/activate

    echo "Virtual environment not found: $VENV_PATH"
    # echo 用来往终端打印文字。

    echo "Set it with: VENV_PATH=/path/to/venv ./run_exp1.sh"
    # 提示用户如何指定虚拟环境路径。

    exit 1 # 表示脚本异常退出。
    # 这里不继续执行，因为没有虚拟环境就无法保证 Python 包环境正确。
fi

source "$VENV_PATH/bin/activate" # 激活 Python 虚拟环境。
# 激活后，终端里的 python / pip 会优先使用这个虚拟环境里的版本。
# i.e., 后面的 python main.py 会使用 ~/pyenv 里的 torch、torchvision 等包。

##############
# 默认训练命令 #
##############

# 这里用数组保存命令和参数。
#
# 为什么不用下面这种写法？
# python main.py \
#     # 注释
#     --model resnet18
#
# 因为 shell 里反斜杠 \ 表示“下一行还是同一条命令”。
# 如果在续行中间插入注释，# 后面的内容会被当成注释忽略，
# 很容易导致命令被截断，或者让 --model 被当成一条新的命令执行。
#
# 用数组写法的好处是：
# - 每个参数单独一行，容易看
# - 每个参数前面可以安全地写注释
# - "${CMD[@]}" 会把数组里的每一项按原样传给 python。
CMD=(
    python main.py

    # 使用 resnet18 模型
    --model resnet18

    # 训练 200 个 epoch
    --steps 200

    # 初始学习率是 0.2
    --lr 0.2

    # 如果 PyTorch 能检测到 CUDA GPU，就使用 GPU 训练
    --gpu
)

"${CMD[@]}" 2>&1 | tee "$OUT_FILE"
# 2>&1：
#   把错误输出 stderr 合并到普通输出 stdout。
#   这样无论是正常打印，还是报错信息，都会一起保存到 results_analysis_exp1/run_exp1_out.txt。

# | tee "$OUT_FILE"：
# - | 是管道，把左边命令的输出交给右边命令。
# - tee 会做两件事：
#   - 把输出继续显示在终端上
#   - 同时把输出写入 results_analysis_exp1/run_exp1_out.txt

TRAIN_STATUS=${PIPESTATUS[0]}
# PIPESTATUS 是 bash 提供的数组变量。
# 上面那一行实际是：
#   python main.py ... | tee results_analysis_exp1/run_exp1_out.txt
#
# 这是一条“管道命令”，有两个程序：
# - 左边：python main.py ...
# - 右边：tee "$OUT_FILE"
#
# PIPESTATUS[0] 保存左边 python main.py 的退出状态。
# - 0 表示训练程序正常结束；
# - 非 0 表示训练程序报错或被中断。
#
# 这里先把它保存到 TRAIN_STATUS，
# 避免后面运行别的命令后 PIPESTATUS 被覆盖。

########################################################
# 训练结束后，自动从 results_analysis_exp1/run_exp1_out.txt 提取每个 epoch 的结果 #
########################################################

echo "Analyzing training log..."
# 在终端提示：开始分析日志。

python "$RESULTS_DIR/analyze_exp1_log.py"
# 运行日志分析脚本。
# 这个脚本本身也放在 results_analysis_exp1/ 里。
#
# 它会读取 results_analysis_exp1/run_exp1_out.txt，并生成：
# - results_analysis_exp1/exp1_epoch_metrics.csv：每个 epoch 一行，记录 train/test loss 和 accuracy；
# - results_analysis_exp1/exp1_accuracy_curve.png：训练准确率和测试准确率曲线；
# - results_analysis_exp1/exp1_loss_curve.png：训练 loss 和测试 loss 曲线。
#
# 如果当前 Python 环境没有安装 matplotlib，脚本会自动改成生成：
# - results_analysis_exp1/exp1_accuracy_curve.svg
# - results_analysis_exp1/exp1_loss_curve.svg
#
# SVG 也是图片格式，可以直接放在 README 里显示。
#
# 老师说“记录训练精度”，通常就是要记录每个 epoch 的 train accuracy。
# 这个 CSV 里对应列名是 train_acc。

exit "$TRAIN_STATUS"
# 最后用训练程序本身的退出状态作为整个脚本的退出状态。
# 这样如果 main.py 训练失败，run_exp1.sh 也会显示失败；
# 但只要日志已经写出，前面的分析脚本仍然会尽量帮你分析已有日志。
