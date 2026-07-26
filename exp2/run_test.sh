#!/usr/bin/env bash

# 对一张真实图片运行 AlexNet 推理，并把预测类别和置信度写到结果图片上。
# 脚本会检查 weights/alexnet 里的所有 .pt 文件，并按照 checkpoint 内部保存的
# 测试准确率 acc 自动选择最佳权重，而不是依赖某个写死的文件名。

set -euo pipefail

# 获取脚本所在的 exp2 目录。这样从任何目录调用本脚本都能找到相关文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认使用 ~/pyenv；也可以在运行时指定其他虚拟环境：
# VENV_PATH=/your/venv/path ./run_test.sh
VENV_PATH="${VENV_PATH:-$HOME/pyenv}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Virtual environment not found: $VENV_PATH" >&2
    echo "Run with: VENV_PATH=/your/venv/path ./run_test.sh" >&2
    exit 1
fi

# 激活安装了 PyTorch、torchvision 和 Pillow 的 Python 环境。
source "$VENV_PATH/bin/activate"

# 老师要求在 CUDA 环境运行，因此这里明确检查 GPU，避免意外使用 CPU。
python -c "import torch, sys; ok=torch.cuda.is_available(); print('CUDA GPU:', torch.cuda.get_device_name(0) if ok else 'unavailable'); sys.exit(0 if ok else 1)" || {
    echo "A CUDA-capable NVIDIA GPU is required." >&2
    exit 1
}

cd "$SCRIPT_DIR"

# 默认读取 test.jpeg，输出 test_result.jpeg，不覆盖原始网络图片。
# test.py 会自动从 weights/alexnet 中选择 checkpoint 内 acc 最大的文件。
python test.py \
    --image "$SCRIPT_DIR/test.jpeg" \
    --weights-dir "$SCRIPT_DIR/weights/alexnet" \
    --output "$SCRIPT_DIR/test_result.jpeg"
