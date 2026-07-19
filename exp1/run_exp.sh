#!/usr/bin/env bash

# Run the ResNet18 CIFAR-10 experiment first in FP32 and then in FP16.
# Each run gets its own log, CSV metrics and figures under results_analysis/.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/pyenv}"
RESULTS_DIR="$SCRIPT_DIR/results_analysis"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Virtual environment not found: $VENV_PATH"
    echo "Set it with: VENV_PATH=/path/to/venv ./run_exp.sh"
    exit 1
fi

source "$VENV_PATH/bin/activate"

python -c "import torch, sys; ok=torch.cuda.is_available(); print('CUDA GPU:', torch.cuda.get_device_name(0) if ok else 'unavailable'); sys.exit(0 if ok else 1)" || {
    echo "The FP32/FP16 timing comparison requires an NVIDIA CUDA GPU."
    exit 1
}

run_training() {
    local precision="$1"
    local out_file="$RESULTS_DIR/run_exp1_out_${precision}.txt"
    local cmd=(
        python main.py
        --model resnet18
        --steps 200
        --lr 0.2
        --gpu
    )

    if [ "$precision" = "fp16" ]; then
        cmd+=(--fp16 --loss_scaling)
    fi

    echo "============================================================"
    echo "Starting experiment 1 ${precision^^} training"
    echo "Log: $out_file"
    echo "============================================================"

    "${cmd[@]}" 2>&1 | tee "$out_file"
    local train_status=${PIPESTATUS[0]}

    echo "Analyzing ${precision^^} training log..."
    python "$RESULTS_DIR/analyze_exp1_log.py" --precision "$precision"
    local analysis_status=$?

    if [ "$train_status" -ne 0 ]; then
        echo "${precision^^} training failed with status $train_status."
        return "$train_status"
    fi
    if [ "$analysis_status" -ne 0 ]; then
        echo "${precision^^} log analysis failed with status $analysis_status."
        return "$analysis_status"
    fi

    echo "Metrics: $RESULTS_DIR/exp1_epoch_metrics_${precision}.csv"
}

# The requested comparison order: finish FP32 before starting FP16.
run_training fp32 || exit $?
run_training fp16 || exit $?

echo "FP32 and FP16 training completed successfully."
