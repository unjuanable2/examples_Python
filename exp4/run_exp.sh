#!/usr/bin/env bash

# exp4：YOLOv3 结构化通道剪枝与视频检测对比总控脚本。
#
# 脚本把作业拆成六个可以单独执行的阶段：
#   ./run_exp.sh download  下载 COCO 2014 的 5k 验证所需图片和 YOLO 标签；
#   ./run_exp.sh original  用原始 YOLOv3 对同一视频做 detect；
#   ./run_exp.sh prune     从老师给的稀疏模型生成 50% 结构化剪枝模型；
#   ./run_exp.sh pruned    用刚生成的 50% 剪枝模型做 detect；
#   ./run_exp.sh compare   用 FFmpeg 把两份结果左右合并；
#   ./run_exp.sh all       按上述顺序执行全部阶段。
#
# 推荐在 GPU Linux 设备上执行。Mac 和 GPU 设备通过 Git 只同步本脚本和源码，
# COCO 数据、模型权重及输出视频由 .gitignore 排除，不通过 Git 传输。

# -u：读取未定义变量时立即报错，避免变量名拼错后继续执行危险的空路径操作。
set -u

# pipefail：管道中任意命令失败都视为失败。例如 python | tee 中 python 报错时，
# 不会因为 tee 成功而把整个阶段误判为成功。
set -o pipefail

# BASH_SOURCE[0] 是本脚本自身路径。转换成绝对目录后，无论用户从哪里调用脚本，
# 后续的 cfg、weights、data 和 output 路径都以 exp4 为基准。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 第一位置参数决定执行阶段；用户不传参数时只显示帮助，不擅自下载或运行长任务。
ACTION="${1:-help}"

# 下列变量均可在命令前临时覆盖。例如：
#   VIDEO=data/samples/my_video.mp4 DEVICE=0 BATCH_SIZE=8 ./run_exp.sh all
# ${变量:-默认值} 表示：环境中没有指定该变量时采用冒号后的默认值。
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-416}"
CONF_THRES="${CONF_THRES:-0.3}"
NMS_THRES="${NMS_THRES:-0.5}"
VIDEO="${VIDEO:-data/samples/c_test.mp4}"

# 本实验固定配对的 cfg 和权重。cfg 决定每层通道数，weights 保存参数值；
# 原模型 cfg 不能加载剪枝权重，剪枝 cfg 也不能加载原模型权重。
ORIGINAL_CFG="cfg/yolov3.cfg"
ORIGINAL_WEIGHTS="weights/yolov3-full-mAP53.3.weights"
SPARSE_WEIGHTS="weights/sparse-yolov3-full-mAP48.1.pt"
PRUNED_CFG="cfg/prune_0.5_yolov3.cfg"
PRUNED_WEIGHTS="weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights"

# 所有生成结果分别保存，防止 detect.py 删除同名输出目录时覆盖另一组实验。
ORIGINAL_OUTPUT="output_original"
PRUNED_OUTPUT="output_pruned50"
RESULTS_DIR="results_analysis"
COMPARISON_VIDEO="$RESULTS_DIR/comparison_prune50.mp4"

# COCO 数据只下载到 exp4/data/coco。5k.txt 会含当前 GPU 设备的绝对路径，
# 所以不能从一台机器直接复制到路径不同的另一台机器。
COCO_DIR="data/coco"
COCO_IMAGES_DIR="$COCO_DIR/images"
COCO_VAL_DIR="$COCO_IMAGES_DIR/val2014"
COCO_LIST="$COCO_DIR/5k.txt"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

# 打印统一格式的阶段标题，让保存的日志更容易阅读。
section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# 发现必要文件不存在时，给出明确提示并终止当前阶段。
require_file() {
    local path="$1"
    local explanation="$2"
    if [ ! -f "$path" ]; then
        echo "Missing file: $path"
        echo "$explanation"
        exit 1
    fi
}

# 检查 Python、PyTorch、OpenCV 和设备选择。DEVICE=cpu 时允许无 CUDA；
# DEVICE=0 时要求 PyTorch 能看到至少一块 NVIDIA CUDA GPU。
check_runtime() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
        echo "Python executable not found: $PYTHON_BIN"
        exit 1
    }

    "$PYTHON_BIN" -c "import cv2, torch; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())" || {
        echo "Python dependencies are incomplete. See README.md section '安装运行环境'."
        exit 1
    }

    if [ "$DEVICE" != "cpu" ]; then
        "$PYTHON_BIN" -c "import torch, sys; ok=torch.cuda.is_available(); print('CUDA GPU:', torch.cuda.get_device_name(0) if ok else 'unavailable'); sys.exit(0 if ok else 1)" || {
            echo "DEVICE=$DEVICE requests CUDA, but CUDA is unavailable."
            echo "Install a CUDA-enabled PyTorch build, or use DEVICE=cpu for detect-only testing."
            exit 1
        }
    fi
}

# 下载 COCO 2014 validation 图片、Darknet/YOLO 格式标签和 5k 验证列表。
# wget -c 支持断点续传；已下载一部分时再次运行不会从零开始。
download_coco() {
    section "Stage 1/5: Download COCO 2014 validation data"

    command -v wget >/dev/null 2>&1 || {
        echo "wget is required. On Ubuntu: sudo apt-get install wget unzip"
        exit 1
    }
    command -v unzip >/dev/null 2>&1 || {
        echo "unzip is required. On Ubuntu: sudo apt-get install unzip"
        exit 1
    }

    mkdir -p "$COCO_IMAGES_DIR"

    # val2014.zip 约 6 GB，包含 41k 张验证图片；5k.part 从中选择本实验的 5000 张。
    if [ ! -d "$COCO_VAL_DIR" ]; then
        wget -c -P "$COCO_IMAGES_DIR" \
            http://images.cocodataset.org/zips/val2014.zip
        unzip -q "$COCO_IMAGES_DIR/val2014.zip" -d "$COCO_IMAGES_DIR"
    else
        echo "Validation images already exist: $COCO_VAL_DIR"
    fi

    # labels.tgz 是已转换为 YOLO 文本格式的标注；每行是 class x_center y_center width height，
    # 后四项均相对于图像宽高归一化到 0~1。
    wget -c -P "$COCO_DIR" \
        https://pjreddie.com/media/files/coco/labels.tgz
    wget -c -P "$COCO_DIR" \
        https://pjreddie.com/media/files/coco/5k.part

    if [ ! -d "$COCO_DIR/labels/val2014" ]; then
        tar -xzf "$COCO_DIR/labels.tgz" -C "$COCO_DIR"
    else
        echo "YOLO labels already exist: $COCO_DIR/labels/val2014"
    fi

    # 5k.part 中的路径以 ./ 开头。把它替换为当前 GPU 设备上的绝对 COCO_DIR，
    # 避免从不同工作目录启动 Python 时找不到图片。
    local coco_abs
    coco_abs="$(cd "$COCO_DIR" && pwd)"
    sed "s#^\./#$coco_abs/#" "$COCO_DIR/5k.part" > "$COCO_LIST"

    # 做三个快速完整性检查：列表应有 5000 行、第一张图存在、同名标签存在。
    local line_count first_image first_label
    line_count="$(wc -l < "$COCO_LIST" | tr -d ' ')"
    first_image="$(head -n 1 "$COCO_LIST")"
    first_label="${first_image/images/labels}"
    first_label="${first_label%.*}.txt"

    if [ "$line_count" -ne 5000 ]; then
        echo "Unexpected image count in $COCO_LIST: $line_count (expected 5000)"
        exit 1
    fi
    require_file "$first_image" "COCO image extraction or 5k.part path generation is incorrect."
    require_file "$first_label" "YOLO labels were not extracted into labels/val2014."

    echo "COCO validation data is ready."
    echo "Images: $COCO_VAL_DIR"
    echo "Labels: $COCO_DIR/labels/val2014"
    echo "Validation list: $COCO_LIST ($line_count images)"
}

# 原模型检测。tee 同时在终端显示逐帧 FPS，并把完整输出保存到日志。
test_original() {
    section "Stage 2/5: Detect video with original YOLOv3"
    check_runtime
    require_file "$ORIGINAL_CFG" "The original YOLOv3 cfg was not found."
    require_file "$ORIGINAL_WEIGHTS" "Copy the teacher-provided original weights to exp4/weights/."
    require_file "$VIDEO" "Set VIDEO to an existing MP4/MOV/AVI path."

    "$PYTHON_BIN" detect.py \
        --cfg "$ORIGINAL_CFG" \
        --weights "$ORIGINAL_WEIGHTS" \
        --data data/coco.data \
        --source "$VIDEO" \
        --output "$ORIGINAL_OUTPUT" \
        --img-size "$IMG_SIZE" \
        --conf-thres "$CONF_THRES" \
        --nms-thres "$NMS_THRES" \
        --device "$DEVICE" \
        --model-label "Original YOLOv3" \
        2>&1 | tee "$RESULTS_DIR/detect_original.log"

    local status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] || exit "$status"
}

# 真正执行 50% 结构化通道剪枝。老师已完成稀疏训练，因此这里直接读取 sparse .pt；
# 脚本会在 COCO 5k 上检查剪枝质量，但按老师要求不调用 train.py 做微调。
prune_model() {
    section "Stage 3/5: Prune 50% of eligible YOLOv3 channels"
    check_runtime
    require_file "$COCO_LIST" "Run ./run_exp.sh download on this GPU device first."
    require_file "$ORIGINAL_CFG" "The original YOLOv3 cfg was not found."
    require_file "$SPARSE_WEIGHTS" "Copy the teacher-provided sparse .pt model to exp4/weights/."

    "$PYTHON_BIN" shortcut_prune.py \
        --cfg "$ORIGINAL_CFG" \
        --data data/coco.data \
        --weights "$SPARSE_WEIGHTS" \
        --percent 0.5 \
        --img_size "$IMG_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        2>&1 | tee "$RESULTS_DIR/prune50.log"

    local status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] || exit "$status"
    require_file "$PRUNED_CFG" "Pruning finished without producing the expected cfg. Check prune50.log."
    require_file "$PRUNED_WEIGHTS" "Pruning finished without producing the expected weights. Check prune50.log."
}

# 用新生成的紧凑模型检测同一视频。所有阈值与原模型阶段完全相同，保证公平比较。
test_pruned() {
    section "Stage 4/5: Detect video with the 50% pruned YOLOv3"
    check_runtime
    require_file "$PRUNED_CFG" "Run ./run_exp.sh prune first."
    require_file "$PRUNED_WEIGHTS" "Run ./run_exp.sh prune first."
    require_file "$VIDEO" "Set VIDEO to an existing MP4/MOV/AVI path."

    "$PYTHON_BIN" detect.py \
        --cfg "$PRUNED_CFG" \
        --weights "$PRUNED_WEIGHTS" \
        --data data/coco.data \
        --source "$VIDEO" \
        --output "$PRUNED_OUTPUT" \
        --img-size "$IMG_SIZE" \
        --conf-thres "$CONF_THRES" \
        --nms-thres "$NMS_THRES" \
        --device "$DEVICE" \
        --model-label "Pruned 50% YOLOv3" \
        2>&1 | tee "$RESULTS_DIR/detect_pruned50.log"

    local status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] || exit "$status"
}

# 将两个逐帧对齐的检测结果缩放为相同高度，再用 hstack 左右拼接。
# basename 只取输入视频文件名，适用于 c_test.mp4 和用户自己的 my_video.mp4。
compare_videos() {
    section "Stage 5/5: Combine original and pruned detections"
    command -v ffmpeg >/dev/null 2>&1 || {
        echo "FFmpeg is required. On Ubuntu: sudo apt-get install ffmpeg"
        exit 1
    }

    local video_name original_video pruned_video
    video_name="$(basename "$VIDEO")"
    original_video="$ORIGINAL_OUTPUT/$video_name"
    pruned_video="$PRUNED_OUTPUT/$video_name"

    require_file "$original_video" "Run ./run_exp.sh original first."
    require_file "$pruned_video" "Run ./run_exp.sh pruned first."

    ffmpeg -y \
        -i "$original_video" \
        -i "$pruned_video" \
        -filter_complex \
        "[0:v]scale=-2:540[left];[1:v]scale=-2:540[right];[left][right]hstack=inputs=2[v]" \
        -map "[v]" -c:v libx264 -pix_fmt yuv420p "$COMPARISON_VIDEO"

    echo "Comparison video: $SCRIPT_DIR/$COMPARISON_VIDEO"
}

show_help() {
    cat <<'EOF'
Usage: ./run_exp.sh <stage>

Stages:
  download   Download and prepare COCO 2014 5k validation data
  original   Run detect.py with the original YOLOv3
  prune      Generate the 50% channel-pruned cfg and weights (no fine-tuning)
  pruned     Run detect.py with the generated 50% pruned YOLOv3
  compare    Combine the two detection videos side by side
  all        Run download -> original -> prune -> pruned -> compare

Optional environment variables:
  VIDEO=data/samples/my_video.mp4  input video shared by both models
  DEVICE=0                         CUDA GPU id; use cpu only for detect testing
  BATCH_SIZE=16                    COCO validation batch size during pruning
  IMG_SIZE=416                     identical model input size for both runs
  CONF_THRES=0.3                   identical confidence threshold
  NMS_THRES=0.5                    identical NMS IoU threshold
  PYTHON_BIN=python                Python executable inside the experiment env
EOF
}

# case 根据用户选择只运行对应阶段；all 用 && 串联，前一步失败就不会继续。
case "$ACTION" in
    download) download_coco ;;
    original) test_original ;;
    prune) prune_model ;;
    pruned) test_pruned ;;
    compare) compare_videos ;;
    all)
        download_coco && test_original && prune_model && test_pruned && compare_videos
        ;;
    help|-h|--help) show_help ;;
    *)
        echo "Unknown stage: $ACTION"
        show_help
        exit 2
        ;;
esac
