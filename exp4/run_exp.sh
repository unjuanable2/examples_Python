#!/usr/bin/env bash
# 上面一行叫 shebang。直接执行 ``./run_exp.sh`` 时，操作系统会通过当前 PATH
# 找到 bash 来解释本文件，而不是假定 bash 一定安装在 /bin/bash。

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

# ``set`` 是 Bash 内建命令，用于改变当前脚本的运行规则。
# -u（nounset）：读取未定义变量时立即报错。
# 例如误把 $RESULTS_DIR 写成 $RESULT_DIR 时，脚本不会把空字符串当成目录继续执行。
set -u

# pipefail：默认情况下，``python ... | tee log`` 的退出状态只看最后一个 tee；
# 即使 Python 已经报错，只要 tee 正常写完，管道仍可能返回 0。启用 pipefail 后，
# 管道里任意命令返回非 0，整个管道就返回失败状态。
# 本脚本还会用 PIPESTATUS[0] 单独取得 Python 的退出码，以便原样退出。
set -o pipefail

# BASH_SOURCE[0] 是本脚本自身路径。转换成绝对目录后，无论用户从哪里调用脚本，
# 后续的 cfg、weights、data 和 output 路径都以 exp4 为基准。
# 这一行从里向外理解：
# 1. ${BASH_SOURCE[0]}：当前脚本文件路径；它比 $0 更适合脚本被 source 的情况；
# 2. dirname：去掉 run_exp.sh 文件名，只保留所在目录；
# 3. cd ... && pwd：进入该目录并取得规范化后的绝对路径；
# 4. $(...)：命令替换，把 pwd 输出保存进 SCRIPT_DIR；
# 5. 双引号：防止路径中包含空格时被 Bash 拆成多个参数。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 第一位置参数决定执行阶段；用户不传参数时只显示帮助，不擅自下载或运行长任务。
# $1 表示脚本收到的第一个位置参数。例如 ``./run_exp.sh prune`` 中 $1 是 prune。
# ${1:-help} 表示：$1 不存在或为空时使用 help，所以误运行 ``./run_exp.sh``
# 只会显示帮助，不会直接开始下载 6 GB 数据或运行长时间验证。
ACTION="${1:-help}"

# 下列变量均可在命令前临时覆盖。例如：
#   VIDEO=data/samples/my_video.mp4 DEVICE=0 BATCH_SIZE=8 ./run_exp.sh all
# ${变量:-默认值} 表示：环境中没有指定该变量时采用冒号后的默认值。
# 使用哪个 Python，默认是当前虚拟环境中的 ``python``。
# 如果环境命令叫 python3，可执行 ``PYTHON_BIN=python3 ./run_exp.sh original``。
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
# 推理设备，默认字符串 0 表示第一块 GPU（cuda:0）；cpu 表示强制使用 CPU。
# DEVICE 作为文本原样传给 detect.py/shortcut_prune.py 的 --device 参数。
BATCH_SIZE="${BATCH_SIZE:-16}"
# 16 是 COCO 5k 验证阶段的默认 batch size。显存不足时可调小。
# 视频检测仍然逐帧、batch size 为 1
IMG_SIZE="${IMG_SIZE:-416}"
# 原模型和剪枝模型共同使用的正方形输入尺寸，默认 416 像素。
# 保持两次相同很重要，否则速度差异可能来自分辨率，而不是剪枝。
CONF_THRES="${CONF_THRES:-0.3}"
# 两次视频检测共同使用的置信度阈值，低于该值的检测框会被丢弃。
# 默认 0.3
NMS_THRES="${NMS_THRES:-0.5}"
# 两次视频检测共同使用的 NMS IoU 阈值，默认 0.5。
# 同类别候选框的重叠 IoU 高于该值时，低置信度的重复框会被抑制。
VIDEO="${VIDEO:-data/samples/c_test.mp4}"
# 两个模型共同处理的测试视频，用户可替换为自己的视频。
# 相对路径会在脚本 cd 到 exp4 后解析，所以默认实际指向 exp4/data/samples/c_test.mp4。

# 本实验固定配对的 cfg 和权重。cfg 是网络“结构图”，weights 是该结构的参数值。
# 原模型 cfg 不能加载剪枝权重：剪枝后 filters/channel 数量不同，tensor shape 对不上。
ORIGINAL_CFG="cfg/yolov3.cfg"
ORIGINAL_WEIGHTS="weights/yolov3-full-mAP53.3.weights"
# 老师提供的稀疏训练模型，只有 prune 阶段使用。
# 稀疏训练已将很多 BN gamma 压到接近 0，但此时网络物理通道数还没有减少。
SPARSE_WEIGHTS="weights/sparse-yolov3-full-mAP48.1.pt"
# 下面两个文件是 ``shortcut_prune.py --percent 0.5`` 的预期输出。
# PRUNED_CFG 记录各层剪枝后的 filters；PRUNED_WEIGHTS 只保存保留通道参数。
PRUNED_CFG="cfg/prune_0.5_yolov3.cfg"
PRUNED_WEIGHTS="weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights"

# 所有生成结果放到 results_analysis。注意 detect.py 启动时会先删除 --output
# 指定的整个目录，因此原模型和剪枝模型必须使用两个不同子目录。
RESULTS_DIR="results_analysis"
# 两份检测视频也统一放进 results_analysis，实验生成物不会散落在 exp4 根目录。
ORIGINAL_OUTPUT="$RESULTS_DIR/output_original"
PRUNED_OUTPUT="$RESULTS_DIR/output_pruned50"
# $RESULTS_DIR/... 是变量展开：实际值为 results_analysis/comparison_prune50.mp4。
COMPARISON_VIDEO="$RESULTS_DIR/comparison_prune50.mp4"

# COCO 数据只下载到 exp4/data/coco。5k.txt 会含当前 GPU 设备的绝对路径，
# 所以不能从一台机器直接复制到路径不同的另一台机器。
COCO_DIR="data/coco"
COCO_IMAGES_DIR="$COCO_DIR/images"
COCO_VAL_DIR="$COCO_IMAGES_DIR/val2014"
COCO_LIST="$COCO_DIR/5k.txt"

# 从此处开始，脚本当前工作目录固定为 exp4。后面的 Python 命令可以稳定使用
# cfg/yolov3.cfg 这类相对路径，而不依赖用户启动脚本时所在的位置。
cd "$SCRIPT_DIR"
# -p 会递归创建缺少的父目录；目录已经存在时也不会报错。
mkdir -p "$RESULTS_DIR"

# 打印统一格式的阶段标题，让保存的日志更容易阅读。
section() {
    # 函数内部的 $1 是传给 section 的第一个参数，不是 run_exp.sh 的全局 $1。
    # 单独 echo 一次空行，让多个阶段在 terminal 和日志中容易区分。
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# 发现必要文件不存在时，给出明确提示并终止当前阶段。
require_file() {
    # local 创建函数局部变量，函数返回后不会覆盖脚本外面同名变量。
    # $1 是要检查的路径，$2 是缺失时给用户看的处理建议。
    local path="$1"
    local explanation="$2"
    # [ ... ] 是 test 命令的传统写法；-f 表示路径存在且是普通文件。
    # ! 对判断取反，所以 ``! -f`` 表示文件不存在。
    if [ ! -f "$path" ]; then
        echo "Missing file: $path"
        echo "$explanation"
        exit 1
    fi
}

# 检查 Python、PyTorch、OpenCV 和设备选择
check_runtime() {
    # command -v 在 PATH 中查找命令。标准输出和错误输出都重定向到 /dev/null，
    # 因为这里只关心退出码：0=找到，非0=找不到。
    # ``|| { ...; }`` 表示左侧失败时执行花括号里的错误处理。
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
        echo "Python executable not found: $PYTHON_BIN"
        exit 1
    }

    # python -c 直接执行后面的短 Python 字符串。能成功 import 表明当前 Python
    # 环境至少安装了 detect.py 必需的 PyTorch 和 OpenCV，同时打印版本方便复现实验。
    "$PYTHON_BIN" -c "import cv2, torch; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())" || {
        echo "Python dependencies are incomplete. See README.md section '安装运行环境'."
        exit 1
    }

    # != 是字符串不相等判断。DEVICE=cpu 时允许无 CUDA；其它值（如 0）要求 CUDA。
    if [ "$DEVICE" != "cpu" ]; then
        # Python 里 ok 保存 CUDA 可用状态；sys.exit(0/1) 把检查结果传回 Bash。
        # 这里只打印第 0 块卡的名称；实际编号仍由 --device "$DEVICE" 传入模型代码。
        "$PYTHON_BIN" -c "import torch, sys; ok=torch.cuda.is_available(); print('CUDA GPU:', torch.cuda.get_device_name(0) if ok else 'unavailable'); sys.exit(0 if ok else 1)" || {
            echo "DEVICE=$DEVICE requests CUDA, but CUDA is unavailable."
            echo "Install a CUDA-enabled PyTorch build, or use DEVICE=cpu for detect-only testing."
            exit 1
        }
    fi
}

# 下载 COCO 2014 validation 图片、Darknet/YOLO 格式标签和 5k 验证列表。
download_coco() {
    # 打印统一格式的阶段标题
    section "Stage 1/5: Download COCO 2014 validation data"

    # 检查 wget 和 unzip 是否可用
    command -v wget >/dev/null 2>&1 || {
        echo "wget is required. On Ubuntu: sudo apt-get install wget unzip"
        exit 1
    }
    command -v unzip >/dev/null 2>&1 || {
        echo "unzip is required. On Ubuntu: sudo apt-get install unzip"
        exit 1
    }

    # COCO_IMAGES_DIR 展开为 data/coco/images。只下载验证集，不创建 train2014。
    mkdir -p "$COCO_IMAGES_DIR"

    # -d 判断目录是否存在。存在 val2014 就跳过 6 GB 图片下载，方便重复运行脚本。
    # 这个判断只代表目录存在；函数末尾还会用第一张图片做实际完整性检查。
    if [ ! -d "$COCO_VAL_DIR" ]; then
        wget -c -P "$COCO_IMAGES_DIR" \
            http://images.cocodataset.org/zips/val2014.zip
        # wget -c 支持断点续传，已下载一部分时再次运行不会从零开始。
        # -P 指定下载目录，避免 wget 把 zip 文件散落在 exp4 根目录。
        
        # val2014.zip 约 6 GB，包含 41k 张验证图片, 5k.part 从中选择本实验的 5000 张。
    
        unzip -q "$COCO_IMAGES_DIR/val2014.zip" -d "$COCO_IMAGES_DIR"
        # unzip -q 解压 zip 文件；
        # -q 是静默模式，避免打印约 41k 张图片的解压日志；
        # -d 指定解压目标目录，结果成为 data/coco/images/val2014/。
    else
        echo "Validation images already exist: $COCO_VAL_DIR"
    fi

    # 这里即使文件已经存在也继续调用 wget -c；wget 会检查已有大小并续传/确认完成。
    wget -c -P "$COCO_DIR" \
        https://pjreddie.com/media/files/coco/labels.tgz
        # labels.tgz 是已转换为 YOLO 文本格式的标注，每行是 class x_center y_center width height，
        # 后四项均相对于图像宽高归一化到 0~1。
    wget -c -P "$COCO_DIR" \
        https://pjreddie.com/media/files/coco/5k.part

    # labels.tgz 使用 gzip 压缩的 tar 归档：
    # -x=解包，-z=先用 gzip 解压，-f=下一参数是归档文件，-C=解到指定目录。
    if [ ! -d "$COCO_DIR/labels/val2014" ]; then
        tar -xzf "$COCO_DIR/labels.tgz" -C "$COCO_DIR"
    else
        echo "YOLO labels already exist: $COCO_DIR/labels/val2014"
    fi

    # 5k.part 中的路径以 ./ 开头。把它替换为当前 GPU 设备上的绝对 COCO_DIR，避免从不同工作目录启动 Python 时找不到图片。
    local coco_abs
    # local 的意思是 coco_abs 只在 download_coco() 函数内有效，函数返回后不会覆盖脚本外面同名变量。
    coco_abs="$(cd "$COCO_DIR" && pwd)"
    # coco_abs 必须在 cd 到 COCO_DIR 的子 shell 中通过 pwd 得到绝对路径。
    # 括号里的 cd 不会改变主脚本当前目录；命令替换只取回 pwd 的输出。
    sed "s#^\./#$coco_abs/#" "$COCO_DIR/5k.part" > "$COCO_LIST"
    # sed 的 s#旧#新# 使用 # 作为分隔符，避免绝对路径中的 / 需要大量转义；
    # ^\./ 只匹配每行开头的 "./"。> 会新建或覆盖本机的 5k.txt。
    

    # 做三个快速完整性检查：列表应有 5000 行、第一张图存在、同名标签存在。
    local line_count first_image first_label
    line_count="$(wc -l < "$COCO_LIST" | tr -d ' ')"
    # wc -l 统计行数；输入重定向 < 避免输出中带文件名；tr -d 删除对齐空格。
    first_image="$(head -n 1 "$COCO_LIST")"
    # head -n 1 读取验证列表第一条绝对图片路径。
    first_label="${first_image/images/labels}"
    first_label="${first_label%.*}.txt"
    # ${变量/旧/新} 是 Bash 字符串替换：把路径中的 images 替换为 labels；
    # ${变量%.*} 删除最后一个点号及扩展名，再拼接 .txt，得到同名 YOLO 标签路径。
    
    
    if [ "$line_count" -ne 5000 ]; then
        # -ne 是整数“不等于”。5k.part 正常应恰好列出 5000 张图片。
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

    # 用 Bash 数组保存命令。数组的每个元素都是一个独立参数，比拼接长字符串更安全：
    # 即使 VIDEO 路径含空格，"${cmd[@]}" 展开时仍不会把它错误拆成多个参数。
    local cmd=(
        "$PYTHON_BIN" detect.py
        # --cfg 描述原始 YOLOv3 每层类型、filters、shortcut、route 和检测头。
        --cfg "$ORIGINAL_CFG"
        # --weights 是与原 cfg shape 对应的 Darknet 二进制参数。
        --weights "$ORIGINAL_WEIGHTS"
        # --data 在普通视频检测里主要用于找到 data/coco.names 类别名称。
        --data data/coco.data
        # --source 是原模型和剪枝模型必须共同使用的输入视频。
        --source "$VIDEO"
        # detect.py 会删除并重建该输出目录；这里专属于原模型。
        --output "$ORIGINAL_OUTPUT"
        # --img-size 是送入网络的尺寸，不会改变最终输出视频原始宽高。
        --img-size "$IMG_SIZE"
        # 低于 conf 的候选框被过滤；重叠 IoU 高于 nms 的重复框被抑制。
        --conf-thres "$CONF_THRES"
        --nms-thres "$NMS_THRES"
        # --device 最终由 torch_utils.select_device() 转为 cuda:0 或 cpu。
        --device "$DEVICE"
        # --model-label 只影响视频左上角文字，不参与模型计算。
        --model-label "Original YOLOv3"
    )

    # "${cmd[@]}" 按数组元素边界执行完整命令。
    # 2>&1 把 stderr 文件描述符 2 重定向到 stdout 文件描述符 1；tee 因而能把
    # 正常日志和 Python traceback 一起显示并保存。
    "${cmd[@]}" 2>&1 | tee "$RESULTS_DIR/detect_original.log"

    # PIPESTATUS 是 Bash 保存上一条管道中每个命令退出码的数组：
    # [0] 对应 Python，[1] 对应 tee。local status=... 必须紧跟管道，否则会被后续命令覆盖。
    local status=${PIPESTATUS[0]}
    # [ "$status" -eq 0 ] 成功时什么也不做；失败时 || 执行 exit，并保留 Python 错误码。
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

    # 这里加载的是 sparse .pt，而不是原始 .weights。只有经过稀疏训练后，BN gamma
    # 才会大量靠近 0，按 gamma 大小剪枝才有“重要性”依据。
    local cmd=(
        "$PYTHON_BIN" shortcut_prune.py
        # 原始 cfg 提供剪枝前的完整层结构。
        --cfg "$ORIGINAL_CFG"
        # coco.data 的 valid 字段告诉 test.py 去哪里读取 COCO 5k。
        --data data/coco.data
        # sparse .pt 保存被 L1 正则压缩过的 BN gamma，是剪枝排序的输入。
        --weights "$SPARSE_WEIGHTS"
        # percent=0.5 用全部可剪枝 BN gamma 排序的第 50% 位置作为全局阈值。
        --percent 0.5
        # 剪枝前后 mAP 和随机输入计时都使用相同 416 网络输入。
        --img_size "$IMG_SIZE"
        # batch-size 只控制 test.py 一次送多少张 COCO 图片，不改变剪枝比例。
        --batch-size "$BATCH_SIZE"
        --device "$DEVICE"
    )

    "${cmd[@]}" 2>&1 | tee "$RESULTS_DIR/prune50.log"

    # 同原模型阶段一样，单独检查管道中 shortcut_prune.py 的退出状态。
    local status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] || exit "$status"
    # Python 返回 0 后仍检查两个最终产物，避免脚本逻辑提前结束却被当成成功。
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

    # 参数结构刻意与 test_original() 保持一致，只替换 cfg、weights、output 和标签。
    # 这样两次检测的输入视频、阈值、尺寸和硬件相同，差异主要来自模型剪枝。
    local cmd=(
        "$PYTHON_BIN" detect.py
        --cfg "$PRUNED_CFG"
        --weights "$PRUNED_WEIGHTS"
        --data data/coco.data
        --source "$VIDEO"
        --output "$PRUNED_OUTPUT"
        --img-size "$IMG_SIZE"
        --conf-thres "$CONF_THRES"
        --nms-thres "$NMS_THRES"
        --device "$DEVICE"
        --model-label "Pruned 50% YOLOv3"
    )

    "${cmd[@]}" 2>&1 | tee "$RESULTS_DIR/detect_pruned50.log"

    # 同样保留 detect.py 的真实错误码，不让 tee 掩盖失败。
    local status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] || exit "$status"
}

# 将两个逐帧对齐的检测结果缩放为相同高度，再用 hstack 左右拼接。
# basename 只取输入视频文件名，适用于 c_test.mp4 和用户自己的 my_video.mp4。
compare_videos() {
    section "Stage 5/5: Combine original and pruned detections"
    # 合成阶段不需要 Python/GPU，但需要系统可执行文件 ffmpeg。
    command -v ffmpeg >/dev/null 2>&1 || {
        echo "FFmpeg is required. On Ubuntu: sudo apt-get install ffmpeg"
        exit 1
    }

    # 一行 local 可以同时声明多个仅在本函数有效的变量。
    local video_name original_video pruned_video
    # basename 删除目录，只保留输入文件名：
    # data/samples/my_video.mp4 -> my_video.mp4。
    video_name="$(basename "$VIDEO")"
    # detect.py 在对应输出目录里沿用源视频文件名，所以可由 basename 推导结果路径。
    original_video="$ORIGINAL_OUTPUT/$video_name"
    pruned_video="$PRUNED_OUTPUT/$video_name"

    require_file "$original_video" "Run ./run_exp.sh original first."
    require_file "$pruned_video" "Run ./run_exp.sh pruned first."

    # FFmpeg 参数解释：
    # -y：目标文件已存在时直接覆盖，便于重复实验；
    # 第一个 -i 是左侧原模型，流编号为 0:v；第二个 -i 是右侧剪枝模型，编号 1:v；
    # -filter_complex：声明多个输入共同参与的视频滤镜图；
    # scale=-2:540：高度统一为 540，宽度按原比例自动计算并取编码所需偶数；
    # [left]/[right]/[v]：给中间滤镜结果起标签；
    # hstack=inputs=2：把两路视频按同一时间轴水平拼接；
    # -map "[v]"：只把最终拼接视频流写入输出，不保留输入音频；
    # libx264：编码为 H.264；yuv420p：提高播放器、浏览器和手机兼容性。
    ffmpeg -y \
        -i "$original_video" \
        -i "$pruned_video" \
        -filter_complex \
        "[0:v]scale=-2:540[left];[1:v]scale=-2:540[right];[left][right]hstack=inputs=2[v]" \
        -map "[v]" -c:v libx264 -pix_fmt yuv420p "$COMPARISON_VIDEO"

    echo "Comparison video: $SCRIPT_DIR/$COMPARISON_VIDEO"
}

# 显示帮助信息。cat <<'EOF' ... EOF 叫 here-document：把中间多行文本原样
# 交给 cat 输出。EOF 加单引号后，其中的 $VIDEO 等字符不会被 Bash 展开。
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

# case 类似 Python 的 match/多分支 if：把 ACTION 与每个模式逐一比较。
# ``模式) 命令 ;;`` 中 ;; 表示该分支结束，不再继续匹配后面的分支。
case "$ACTION" in
    # 只执行用户明确选择的单一阶段，方便下载中断后续传或单独重跑某个模型。
    download) download_coco ;;
    original) test_original ;;
    prune) prune_model ;;
    pruned) test_pruned ;;
    compare) compare_videos ;;
    all)
        # && 只有左侧函数返回 0 时才运行右侧函数。因此 download/原模型/剪枝中
        # 任一阶段失败，后续不会继续生成可能误导的结果和对比视频。
        download_coco && test_original && prune_model && test_pruned && compare_videos
        ;;
    # 下面三个写法都显示帮助并正常返回 0。
    help|-h|--help) show_help ;;
    # * 匹配所有未知参数。退出码 2 通常表示命令行使用方式错误，区别于运行失败的 1。
    *)
        echo "Unknown stage: $ACTION"
        show_help
        exit 2
        ;;
esac
