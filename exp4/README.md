- [exp4：YOLOv3 结构化通道剪枝实验](#exp4yolov3-结构化通道剪枝实验)
  - [实验目标](#实验目标)
  - [文件说明](#文件说明)
  - [安装运行环境](#安装运行环境)
  - [实验流程](#实验流程)
  - [分步命令速查](#分步命令速查)
    - [下载并准备 COCO 5k 验证集](#下载并准备-coco-5k-验证集)
    - [测试原模型](#测试原模型)
    - [生成剪枝 50% 的模型](#生成剪枝-50-的模型)
    - [测试剪枝 50% 后的模型](#测试剪枝-50-后的模型)
    - [合成一个对比视频](#合成一个对比视频)
  - [代码执行过程](#代码执行过程)
  - [COCO mAP 和视频检测分别说明什么](#coco-map-和视频检测分别说明什么)
  - [结果记录和分析](#结果记录和分析)
  - [常见问题](#常见问题)

# exp4：YOLOv3 结构化通道剪枝实验

这个实验使用 PyTorch 版 YOLOv3，比较原始模型和结构化通道剪枝 50% 模型的检测质量、模型大小、参数量、COCO mAP 和视频处理速度。
 
## 实验目标

完整实验输出包括：

1. 原 YOLOv3 对同一视频的检测结果；
2. 从稀疏模型实际生成的 50% 通道剪枝 cfg 和 weights；
3. 剪枝模型对同一视频的检测结果；
4. 原模型和剪枝模型的 COCO 5k mAP、参数量和推理时间；
5. 左侧原模型、右侧剪枝模型的合并视频；
6. 对速度提升、文件缩小、置信度变化、误检和漏检的分析。

公平比较必须满足：

- 相同输入视频；
- 相同输入尺寸，默认 `416×416`；
- 相同置信度阈值，默认 `0.3`；
- 相同 NMS IoU 阈值，默认 `0.5`；
- 相同 GPU、PyTorch 环境和精度模式；
- 视频推理都使用 batch size 1。

## 文件说明

```text
exp4/
├── README.md                         # 本文：实验原理、操作步骤和结果解释
├── run_exp.sh                        # 推荐入口：分阶段下载、剪枝、检测和合成视频
├── requirements.txt                  # 除 PyTorch 外的 Python 依赖
├── detect.py                         # 图片/视频推理，画检测框、模型名称和 FPS
├── shortcut_prune.py                 # 按 BN gamma 对 shortcut 相关通道剪枝
├── test.py                           # 使用 COCO 5k 标签计算 P、R、mAP 和 F1
├── train.py                          # 训练/稀疏训练/微调；本作业不执行
├── models.py                         # 解析 cfg、建立 Darknet/YOLOv3、读写权重
├── cfg/
│   ├── yolov3.cfg                    # 原始 YOLOv3 网络结构
│   ├── prune_0.6_yolov3.cfg          # 老师给的 60% 剪枝参考结构
│   └── prune_0.5_yolov3.cfg          # 执行 prune 后生成，不应手工编写
├── weights/
│   ├── yolov3-full-mAP53.3.weights   # 原模型，约 248 MB
│   ├── sparse-yolov3-full-mAP48.1.pt # 老师给的稀疏模型，剪枝输入
│   ├── prune_0.6_sparse-yolov3-full-mAP48.1.weights
│   │                                  # 老师给的 60% 剪枝参考权重
│   └── prune_0.5_sparse-yolov3-full-mAP48.1.weights
│                                      # 执行 prune 后生成的 50% 剪枝权重
├── data/
│   ├── coco.data                     # 类别数、5k.txt、coco.names 的路径配置
│   ├── coco.names                    # COCO 80 类名称
│   ├── samples/c_test.mp4            # 老师给的示例输入视频
│   └── coco/                         # download 阶段生成，不提交 Git
│       ├── images/val2014/           # COCO 2014 validation 图片
│       ├── labels/val2014/           # YOLO 文本格式 ground truth
│       └── 5k.txt                    # 本机绝对路径组成的 5000 张图片列表
├── utils/                             # 数据读取、NMS、AP 和剪枝辅助函数
├── output_original/                   # 原模型检测视频，不提交 Git
├── output_pruned50/                   # 50% 剪枝模型检测视频，不提交 Git
└── results_analysis/
    ├── prune50.log                   # 剪枝与 COCO mAP 完整日志
    ├── detect_original.log           # 原模型逐帧 FPS 日志
    ├── detect_pruned50.log           # 剪枝模型逐帧 FPS 日志
    └── comparison_prune50.mp4        # 最终左右对比视频
```

三个主要模型的关系：

```text
原始模型
yolov3.cfg + yolov3-full-mAP53.3.weights
                 │
                 │ 老师已经完成稀疏训练
                 ▼
稀疏模型
yolov3.cfg + sparse-yolov3-full-mAP48.1.pt
                 │
                 │ shortcut_prune.py --percent 0.5
                 ▼
50% 结构化剪枝模型
prune_0.5_yolov3.cfg + prune_0.5_sparse-yolov3-full-mAP48.1.weights
```

`.cfg` 描述网络每层结构，`.weights`/`.pt` 保存参数值。原 cfg 和剪枝权重不能混用，因为剪枝后各层通道数已经改变。

`--percent 0.5` 表示在全部可剪枝 BN gamma 的排序中以 50% 位置确定全局阈值，不表示最终参数量或模型文件一定刚好减少 50%。shortcut 层需要保持相加两侧通道一致，也会影响实际剪枝率。

## 安装运行环境

推荐使用带 NVIDIA GPU 的 Linux 设备。Mac 可以运行 `detect.py --device cpu` 检查流程，但这份旧代码不支持 Apple MPS，跑 COCO 5k 会很慢。

进入 GPU 设备的仓库：

```bash
git pull
cd intern1/exp4
```

创建独立环境，示例使用 Python 3.10：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与 GPU 驱动匹配的 CUDA PyTorch，再安装项目其它依赖：

```bash
python -m pip install -r requirements.txt
```

检查环境：

```bash
python -c "import torch, cv2; print('torch:', torch.__version__); print('opencv:', cv2.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

输出中的 `CUDA` 应为 `True`。如果 GPU 显存不足，可在运行脚本时降低 COCO 验证 batch size：

```bash
BATCH_SIZE=8 ./run_exp.sh prune
```

COCO、模型权重、输出视频已经写入 `.gitignore`。Mac 与 GPU 设备通过 Git 同步代码即可；老师给的权重应使用 `scp`、网盘或移动硬盘单独传到 GPU 设备的 `exp4/weights/`，不要直接 `git add .` 提交数百 MB 权重。

## 实验流程

实验流程可以按代码实际执行顺序理解：

- `run_exp.sh` 是整个实验的推荐入口，负责读取用户选择的阶段、准备公共参数，并依次调用下载命令、`detect.py`、`shortcut_prune.py` 和 FFmpeg。
  ```bash
  # 第一次在 GPU 设备上建议分阶段执行，方便发现数据、权重或环境问题
  ./run_exp.sh download
  ./run_exp.sh original
  ./run_exp.sh prune
  ./run_exp.sh pruned
  ./run_exp.sh compare

  # 上面各阶段确认正常后，也可以一次执行完整流程
  ./run_exp.sh all
  ```
  - `download`：只准备 COCO 2014 的 5k 验证数据，不下载训练集；
  - `original`：用原始 YOLOv3 检测视频；
  - `prune`：从老师给的稀疏模型生成 50% 通道剪枝模型，并在 COCO 5k 上验证；
  - `pruned`：用新生成的 50% 剪枝模型检测同一个视频；
  - `compare`：把原模型和剪枝模型的输出按帧左右合并；
  - `all`：严格按照 `download → original → prune → pruned → compare` 的顺序执行。前一阶段失败时后续阶段不会继续，避免拿错误模型生成最终视频。
- `run_exp.sh` 开头先准备所有阶段共用的配置：
  - `SCRIPT_DIR`：`run_exp.sh` 所在的 `exp4` 绝对路径；脚本随后执行 `cd "$SCRIPT_DIR"`，所以从仓库根目录或其他目录启动都能找到 cfg 和 weights；
  - `PYTHON_BIN`：使用哪个 Python，默认是当前虚拟环境中的 `python`；
  - `DEVICE`：推理设备，默认 `0` 表示第 0 块 CUDA GPU；
  - `BATCH_SIZE=16`：只用于 COCO mAP 验证；视频检测仍然逐帧、batch size 为 1；
  - `IMG_SIZE=416`：原模型和剪枝模型共同使用的输入尺寸；
  - `CONF_THRES=0.3`、`NMS_THRES=0.5`：两次视频检测共同使用的置信度和 NMS 阈值；
  - `VIDEO=data/samples/c_test.mp4`：两个模型共同处理的视频。
  - 这些变量使用 `${变量:-默认值}` 形式，因此可以在命令前临时覆盖。例如：
    ```bash
    VIDEO=data/samples/my_video.mp4 BATCH_SIZE=8 DEVICE=0 ./run_exp.sh all
    ```
    - `VIDEO=...` 只对这次命令生效，不需要修改 Python 源码；
    - `BATCH_SIZE=8` 可降低 COCO 验证的显存占用；
    - 原模型和剪枝模型必须传相同的 `VIDEO/IMG_SIZE/CONF_THRES/NMS_THRES/DEVICE`，否则速度与检测质量不具备公平可比性。
- 当阶段是 `download` 时，`run_exp.sh` 调用 `download_coco()` 准备 COCO 5k 验证数据：
  - 先通过 `command -v wget` 和 `command -v unzip` 检查下载及解压工具；缺少工具时立即退出，而不是下载到一半才失败；
  - `mkdir -p data/coco/images` 创建数据目录；`-p` 表示目录已经存在时不报错；
  - `wget -c` 下载 `val2014.zip`：
    - `-c` 表示断点续传，网络中断后重新运行 `download` 可以继续；
    - COCO 2014 validation 压缩包约 6 GB、含约 41k 张图片；本实验只通过 `5k.part` 从中选 5000 张验证图片；
    - 不下载约 13 GB 的 `train2014`，因为老师已经提供稀疏模型，本实验不训练也不微调；
  - `unzip` 把图片解压到 `data/coco/images/val2014/`；
  - 下载 `labels.tgz` 并由 `tar` 解压到 `data/coco/labels/val2014/`；这些是 Darknet 已转换好的 YOLO ground-truth 标签；
  - 每张图片与标签按文件名一一对应：
    ```text
    data/coco/images/val2014/COCO_val2014_000000123456.jpg
    data/coco/labels/val2014/COCO_val2014_000000123456.txt
    ```
    - 标签每行是 `class_id x_center y_center width height`；
    - `class_id` 是 COCO 80 类中的类别编号；
    - 坐标和宽高相对于图片尺寸归一化到 `[0,1]`；
  - 下载 `5k.part` 后，`sed` 把其中以 `./` 开头的相对路径转换成当前 GPU 设备的绝对路径，生成 `data/coco/5k.txt`；
    - 使用绝对路径可以让 `test.py` 无论从哪个工作目录启动都找到图片；
    - 绝对路径和机器目录有关，所以 `5k.txt` 不应该从 Mac 提交给 GPU 设备，而应在 GPU 设备重新生成；
  - 最后检查 `5k.txt` 是否恰好 5000 行，以及第一张图片和同名标签是否存在。只有三个检查都通过，download 阶段才算成功。
- `data/coco.data` 把下载结果交给 Python 程序：
  - `classes=80`：模型需要检测 80 个 COCO 类别；
  - `valid=data/coco/5k.txt`：`test.py` 根据这个列表加载 5000 张验证图片；
  - `names=data/coco.names`：把模型输出的类别编号转成 `person`、`car` 等文字；
  - `train` 在本作业中不会使用，因为没有训练和微调；它暂时也指向 `5k.txt`，只是避免保留老师机器上的 `/opt/DATASET/coco/` 绝对路径。
- 当阶段是 `original` 时，`run_exp.sh` 调用 `test_original()`：
  - `check_runtime()` 先导入 PyTorch 和 OpenCV，打印版本并检查 CUDA；当 `DEVICE=0` 时，如果 `torch.cuda.is_available()` 是 `False` 就立即退出；
  - `require_file()` 检查以下三个输入确实存在：
    ```text
    cfg/yolov3.cfg
    weights/yolov3-full-mAP53.3.weights
    data/samples/c_test.mp4
    ```
  - 随后用 Bash 数组/续行参数实际执行：
    ```bash
    python detect.py \
      --cfg cfg/yolov3.cfg \
      --weights weights/yolov3-full-mAP53.3.weights \
      --data data/coco.data \
      --source data/samples/c_test.mp4 \
      --output output_original \
      --img-size 416 \
      --conf-thres 0.3 \
      --nms-thres 0.5 \
      --device 0 \
      --model-label "Original YOLOv3"
    ```
  - `2>&1 | tee results_analysis/detect_original.log`：
    - `2>&1` 把报错信息和普通输出合并；
    - `tee` 让逐帧 FPS 既显示在 terminal，也保存到日志；
    - `PIPESTATUS[0]` 取得 `python detect.py` 本身的退出状态，避免 Python 失败但 `tee` 成功时误判实验成功。
- `detect.py` 收到原模型参数后开始处理视频：
  - 文件末尾的 `argparse.ArgumentParser()` 定义程序支持的所有 `--参数`；
  - `opt = parser.parse_args()` 真正读取 `run_exp.sh` 传入的参数，得到 `opt.cfg`、`opt.weights`、`opt.source` 等属性；
  - `with torch.no_grad()` 关闭梯度记录：目标检测只做前向传播，不需要反向传播，可节省显存和计算；
  - `detect()` 首先调用 `torch_utils.select_device(opt.device)`：
    - `--device 0` 返回 `torch.device('cuda:0')`；
    - `--device cpu` 返回 CPU；
  - 如果 `output_original/` 已存在，旧版代码会先删除整个目录再重新创建，因此原模型和剪枝模型必须使用不同输出目录；
  - `model = Darknet(opt.cfg, img_size)` 根据 `cfg/yolov3.cfg` 创建原始网络对象；
    - `Darknet.__init__()` 在 `models.py` 中；
    - `parse_model_cfg()` 在 `utils/parse_config.py` 中读取 cfg；
    - 它把每一个 `[convolutional]`、`[shortcut]`、`[route]`、`[upsample]`、`[yolo]` 文本块转换成字典；
    - `create_modules()` 再把字典转换成真正的 PyTorch `nn.Conv2d`、`nn.BatchNorm2d`、`nn.Upsample` 和 `YOLOLayer` 对象；
    - 所有层放进 `nn.ModuleList`，其顺序与 cfg 层顺序对应；
  - 权重加载根据文件后缀分两种：
    - `.pt` 用 `torch.load()` 读取 PyTorch checkpoint；
    - `.weights` 用 `models.py` 的 `load_darknet_weights()` 按 Darknet 文件顺序把 BN、卷积 bias 和卷积 weight 复制进模型；
  - `model.to(device).eval()`：
    - `.to(device)` 把参数放进 GPU 显存；
    - `.eval()` 让 BatchNorm 使用已有的 running mean/variance，不再更新训练统计量；
  - `LoadImages(source, img_size=416)` 在 `utils/datasets.py` 中创建视频读取对象：
    - `cv2.VideoCapture()` 打开 MP4/MOV/AVI；
    - 每次循环取出一个原始 BGR 视频帧 `img0`；
    - `letterbox()` 保持宽高比例缩放，并用边框补到适合 YOLOv3 的尺寸；
    - `img[:, :, ::-1]` 把 OpenCV 的 BGR 转为模型使用的 RGB；
    - `.transpose(2,0,1)` 把 `[height,width,channel]` 改成 PyTorch 的 `[channel,height,width]`；
    - `astype(np.float32)` 把整数像素转为浮点数，`img /= 255.0` 把范围从 `[0,255]` 缩放到 `[0,1]`；
  - 对每一帧执行：
    - `torch.from_numpy(img).to(device)` 把 numpy 图像变成 GPU tensor；
    - `unsqueeze(0)` 增加 batch 维度，得到近似 `[1,3,416,416]`；
    - `pred, _ = model(img)` 前向传播，得到三个 YOLO 检测尺度产生的候选框；
    - `non_max_suppression(pred, conf_thres, nms_thres)` 先过滤低置信度框，再删除同类别高度重叠的重复框；
    - `scale_coords()` 把 416 输入坐标映射回原视频分辨率；
    - `plot_one_box()` 画类别名称、置信度和边界框；
    - CUDA 同步后计算 `frame_fps = 1 / frame_seconds`，再用 `cv2.putText()` 把 `Original YOLOv3 | FPS ...` 写到左上角；
    - `cv2.VideoWriter` 使用原视频 FPS 和宽高，把处理后的帧写进 `output_original/c_test.mp4`。
- 当阶段是 `prune` 时，`run_exp.sh` 调用 `prune_model()`：
  - 检查 CUDA、`data/coco/5k.txt`、原始 cfg 和老师提供的稀疏模型；
  - 真正执行：
    ```bash
    python shortcut_prune.py \
      --cfg cfg/yolov3.cfg \
      --data data/coco.data \
      --weights weights/sparse-yolov3-full-mAP48.1.pt \
      --percent 0.5 \
      --img_size 416 \
      --batch-size 16 \
      --device 0
    ```
  - `--percent 0.5` 是全局 BN gamma 排序的阈值位置；它不是把 50% 参数简单改成 0，也不保证权重文件恰好缩小 50%；
  - 完整输出通过 `tee` 保存到 `results_analysis/prune50.log`，这里是报告中 mAP、参数量和前向时间的主要来源。
- `shortcut_prune.py` 的实际执行过程是：
  - `parser.parse_args()` 读取剪枝比例、cfg、稀疏权重、图片尺寸、batch size 和设备；
  - `Darknet(yolov3.cfg)` 先创建通道数尚未减少的完整模型；
  - `torch.load(sparse-yolov3-full-mAP48.1.pt)['model']` 取出老师已经稀疏训练好的模型参数；
  - 稀疏训练的作用不是立即删通道，而是在训练期间给 BN gamma 加 L1 约束，让大量不重要通道的 gamma 靠近 0；老师已完成这一步，本作业不再运行 `train.py -sr`；
  - `parse_module_defs2(model.module_defs)` 遍历 YOLOv3 层定义，返回：
    - `CBL_idx`：卷积 + BN + LeakyReLU 层下标；
    - `Conv_idx`：不带 BN 的卷积层下标，通常是 YOLO 输出层前的卷积；
    - `prune_idx`：允许参与 gamma 排序与剪枝的层；
    - `shortcut_idx/shortcut_all`：残差相加两侧层的对应关系；
  - `gather_bn_weights()`：
    - 访问每个可剪枝 BN 的 `bn_module.weight`，这就是缩放参数 gamma；
    - 取绝对值并拼成一个一维 tensor；
    - gamma 越接近 0，代表该通道缩放后的输出越弱，被认为越不重要；
  - `torch.sort(bn_weights)` 从小到大排序全部 gamma；
  - `thre_index = int(len(sorted_bn) * 0.5)` 找到第 50% 位置，`threshold = sorted_bn[thre_index]` 得到全局门限；
  - `obtain_bn_mask(bn_module, threshold)` 为每层生成由 0/1 组成的 mask：
    - gamma 大于阈值的通道记为 1，保留；
    - gamma 小于阈值的通道记为 0，删除；
    - shortcut 是逐元素相加，两侧 channel 数必须一致，所以 shortcut 关联层使用相互对应的 mask；
  - `prune_and_eval()` 先复制一份完整模型并把低 gamma 通道乘 0：
    - 这一步只让通道输出变为 0，模型物理结构和参数量还没减少；
    - 随后调用 `test.py`，观察“假剪枝/mask 模型”的 COCO mAP；
    - 如果此时 mAP 接近 0，说明 50% 阈值、稀疏程度或 mask 关系可能有问题，不应继续相信生成结果；
  - `obtain_filters_mask()` 计算每个 CBL 层最终保留多少通道，并打印 `total channel / remaining channel`；
  - `prune_model_keep_size2()` 处理被删通道的 BN beta 偏移，得到保持原结构但数值效果接近剪枝后的中间模型；
  - `compact_module_defs = deepcopy(model.module_defs)` 复制 cfg 层定义，再把每层 `filters` 改为实际保留通道数；
  - `compact_model = Darknet(compact_module_defs)` 根据新 filters 创建真正变窄的模型：
    - 这时卷积张量尺寸和参数量才真实减少；
    - 普通 GPU/CPU 可以直接对更小的 dense convolution 加速，不依赖稀疏矩阵硬件；
  - `init_weights_from_loose_model()` 根据每层 mask，把保留的卷积核、BN gamma、beta、running mean 和 running variance 从原模型复制进 compact model；
  - 随机生成 `[1,3,416,416]` 输入，分别多次运行中间模型和 compact model，统计平均前向时间；
  - 再次调用 `test.py` 评估最终 compact model；如果 mask 模型与 compact model mAP 相差异常，可能说明参数复制或 shortcut 通道处理错误；
  - `AsciiTable` 输出剪枝前后 mAP、Parameters 和 Inference；
  - `write_cfg()` 保存 `cfg/prune_0.5_yolov3.cfg`；
  - `save_weights()` 保存 `weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights`；
  - 脚本到这里结束，不调用 `train.py`，因此没有微调。
- `shortcut_prune.py` 每次调用 `test.py` 时，COCO mAP 的计算过程是：
  - `parse_data_cfg(data/coco.data)` 找到 `valid=data/coco/5k.txt`；
  - `LoadImagesAndLabels` 读取 5000 张图片，并将图片路径中的 `images` 替换为 `labels`，寻找同名真实标注；
  - `DataLoader` 按 `--batch-size 16` 组合 batch；显存不足时可以通过 `BATCH_SIZE=8` 降低；
  - 每个 batch 执行模型前向传播和 NMS；
  - 对每张图片：
    - 取预测类别、置信度和预测框；
    - 取真实类别和真实框；
    - 只有预测类别正确、预测框和真实框 IoU 大于 `0.5`，并且该真实框尚未被另一个预测匹配，才把该预测标为正确；
  - `ap_per_class()` 汇总不同置信度位置的 Precision 和 Recall，计算每个类别 PR 曲线下的面积 AP；
  - 对参与评估类别的 AP 取平均得到 mAP；
  - `test.py` 只评估模型，不执行 `loss.backward()` 和 `optimizer.step()`，所以 COCO 验证不会修改模型参数。
- 当阶段是 `pruned` 时，`run_exp.sh` 调用 `test_pruned()`：
  - 使用的 Python 仍然是同一个 `detect.py`，数据读取、前向传播、NMS、画框和视频保存流程也完全相同；
  - 唯一关键变化是模型配对改成：
    ```text
    cfg/prune_0.5_yolov3.cfg
    weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights
    ```
  - 输入仍为同一 `VIDEO`，尺寸仍为 416，置信度/NMS 阈值仍为 0.3/0.5，设备仍为同一 GPU；
  - 视频左上角文字改为 `Pruned 50% YOLOv3 | FPS ...`；
  - 结果保存到 `output_pruned50/`，日志保存到 `results_analysis/detect_pruned50.log`；
  - 两边复用完全相同的推理代码，因此观察到的主要差异来自网络结构和模型权重，而不是两套实现不同。
- 当阶段是 `compare` 时，`run_exp.sh` 调用 `compare_videos()`：
  - `basename "$VIDEO"` 只取得输入文件名，例如 `my_video.mp4`；
  - 分别检查 `output_original/my_video.mp4` 和 `output_pruned50/my_video.mp4`；
  - FFmpeg 的两个 `scale=-2:540` 把视频缩放到相同高度 540，并让宽度自动取合法偶数；
  - `hstack=inputs=2` 按相同时间顺序把原模型放左边、剪枝模型放右边；
  - `libx264 + yuv420p` 生成兼容性较好的 H.264 MP4；
  - 最终保存为 `results_analysis/comparison_prune50.mp4`。
- `run_exp.sh` 全部完成后，实验的主要结果对应关系为：
  - `results_analysis/prune50.log`：回答剪枝前后 mAP、参数量和模型前向时间变化；
  - `results_analysis/detect_original.log`：原模型逐帧 FPS；
  - `results_analysis/detect_pruned50.log`：剪枝模型逐帧 FPS；
  - `output_original/<视频名>`：原模型可视化检测结果；
  - `output_pruned50/<视频名>`：剪枝模型可视化检测结果；
  - `results_analysis/comparison_prune50.mp4`：最终提交的左右对比视频。

把上述步骤压缩成一句话就是：`run_exp.sh` 在 GPU 设备准备 COCO 5k，`detect.py` 用原 cfg/weights 生成基准视频，`shortcut_prune.py` 根据稀疏模型的 BN gamma 排序删除 50% 低重要性通道并由 `test.py` 验证 mAP，再由同一个 `detect.py` 处理同一视频，最后 FFmpeg 将两份结果左右合并；整个过程只做评估和前向传播，不运行 `train.py` 微调。

## 分步命令速查

查看总控脚本支持的阶段：

```bash
chmod +x run_exp.sh
./run_exp.sh --help
```

输出中的六个阶段是：

```text
download  下载并准备 COCO 2014 5k 验证数据
original  用原模型检测视频
prune     从 sparse .pt 生成 50% 剪枝模型，不微调
pruned    用 50% 剪枝模型检测同一视频
compare   左右合并两个结果视频
all       按以上顺序全部执行
```

第一次建议逐阶段运行，某一步出错时容易定位。确认环境和路径都正确后，也可以执行：

```bash
./run_exp.sh all
```

### 下载并准备 COCO 5k 验证集

执行：

```bash
./run_exp.sh download
```

这一阶段具体完成：

1. 从 COCO 官方图片服务器下载约 6 GB 的 `val2014.zip`，`wget -c` 支持断点续传；
2. 解压到 `data/coco/images/val2014/`；
3. 下载 Darknet 提供的 `labels.tgz` 和 `5k.part`；
4. 解压 YOLO 格式标签到 `data/coco/labels/val2014/`；
5. 将 `5k.part` 的相对路径改为当前 GPU 设备的绝对路径，生成 `5k.txt`；
6. 检查列表恰好 5000 行，并确认第一张图片和对应标签都存在。

数据关系如下：

```text
images/val2014/COCO_val2014_000000123456.jpg
labels/val2014/COCO_val2014_000000123456.txt
```

标签每一行格式：

```text
class_id x_center y_center width height
```

后四项除以图片宽高归一化到 `[0,1]`。`test.py` 根据同名图片和标签比较预测框与人工真实框。

只做 5k mAP 不需要下载约 13 GB 的 `train2014`，因为本实验不训练、不微调。

### 测试原模型

先确认原模型存在：

```text
cfg/yolov3.cfg
weights/yolov3-full-mAP53.3.weights
```

执行：

```bash
./run_exp.sh original
```

它等价于：

```bash
python detect.py \
  --cfg cfg/yolov3.cfg \
  --weights weights/yolov3-full-mAP53.3.weights \
  --data data/coco.data \
  --source data/samples/c_test.mp4 \
  --output output_original \
  --img-size 416 \
  --conf-thres 0.3 \
  --nms-thres 0.5 \
  --device 0 \
  --model-label "Original YOLOv3"
```

各参数含义：

- `--cfg`：原模型网络结构；
- `--weights`：原模型参数；
- `--data`：从中读取 `data/coco.names`，把类别编号显示为人、车等名称；
- `--source`：输入视频；
- `--output`：输出目录。`detect.py` 会删除并重建这个目录；
- `--img-size 416`：每帧 letterbox 后输入模型的尺寸；
- `--conf-thres 0.3`：过滤置信度低于 0.3 的预测；
- `--nms-thres 0.5`：NMS 合并重叠框使用的 IoU 阈值；
- `--device 0`：使用第 0 块 CUDA GPU；
- `--model-label`：写在输出视频左上角的模型名称。

结果：

```text
output_original/c_test.mp4
results_analysis/detect_original.log
```

终端和日志逐帧输出同步后的近似端到端 FPS，输出视频左上角也会显示当前 FPS。

使用自己的 iPhone 视频时，将 MOV/MP4 放进 `data/samples/`，然后运行：

```bash
VIDEO=data/samples/my_video.mp4 ./run_exp.sh original
```

源码支持 `.mov`、`.avi` 和 `.mp4`。如果 iPhone HEVC/H.265 无法被 OpenCV 解码，可转为 H.264：

```bash
ffmpeg -i my_video.MOV -c:v libx264 -pix_fmt yuv420p -an data/samples/my_video.mp4
```

### 生成剪枝 50% 的模型

先确认稀疏模型存在：

```text
weights/sparse-yolov3-full-mAP48.1.pt
```

执行：

```bash
./run_exp.sh prune
```

它等价于：

```bash
python shortcut_prune.py \
  --cfg cfg/yolov3.cfg \
  --data data/coco.data \
  --weights weights/sparse-yolov3-full-mAP48.1.pt \
  --percent 0.5 \
  --img_size 416 \
  --batch-size 16 \
  --device 0
```

执行顺序：

1. `Darknet(cfg/yolov3.cfg)` 创建完整 YOLOv3；
2. 加载老师已经稀疏训练好的 `.pt`；
3. `parse_module_defs2()` 找出卷积+BN 层、普通卷积层和 shortcut 对应关系；
4. `gather_bn_weights()` 收集所有可剪枝 BN 的 `|gamma|`；
5. 对 gamma 排序，在 `percent=0.5` 位置取得阈值；
6. 小于阈值的通道 mask 设为 0；shortcut 相连层复用对应 mask，保证张量能相加；
7. 在 COCO 5k 上检查 mask 后模型 mAP；
8. 修改 cfg 中每个卷积层的 `filters`，建立真正减少通道的 compact model；
9. 将保留通道的卷积和 BN 参数复制到 compact model；
10. 测量参数量、前向时间和最终 mAP；
11. 保存新的 cfg 和 weights。

预期生成：

```text
cfg/prune_0.5_yolov3.cfg
weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights
results_analysis/prune50.log
```

日志中的表格会报告：

```text
Metric       Before       After
mAP          剪枝前       剪枝后
Parameters   原参数量     紧凑模型参数量
Inference    原前向时间   紧凑模型前向时间
```

老师要求不微调，所以到这里不执行 `train.py`。精度下降正是本作业需要观察的 detection-quality trade-off，而不是再花时间训练恢复。

### 测试剪枝 50% 后的模型

生成 50% 模型后执行：

```bash
./run_exp.sh pruned
```

它等价于：

```bash
python detect.py \
  --cfg cfg/prune_0.5_yolov3.cfg \
  --weights weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights \
  --data data/coco.data \
  --source data/samples/c_test.mp4 \
  --output output_pruned50 \
  --img-size 416 \
  --conf-thres 0.3 \
  --nms-thres 0.5 \
  --device 0 \
  --model-label "Pruned 50% YOLOv3"
```

结果：

```text
output_pruned50/c_test.mp4
results_analysis/detect_pruned50.log
```

运行自选视频时，必须和原模型阶段传相同 `VIDEO`：

```bash
VIDEO=data/samples/my_video.mp4 ./run_exp.sh pruned
```

不要直接把老师提供的 60% 权重改名为 50% 权重；文件名改变不会改变模型结构。50% cfg 和 weights 必须由 `shortcut_prune.py --percent 0.5` 实际生成。

### 合成一个对比视频

确保两个模型已经处理相同输入，然后执行：

```bash
./run_exp.sh compare
```

自选视频要继续传相同变量：

```bash
VIDEO=data/samples/my_video.mp4 ./run_exp.sh compare
```

FFmpeg 会把两边缩放到相同高度后按帧左右拼接：

```text
┌────────────────────────────┬────────────────────────────┐
│ Original YOLOv3 | FPS ...  │ Pruned 50% YOLOv3 | FPS...│
│ 原模型检测框               │ 剪枝模型检测框             │
└────────────────────────────┴────────────────────────────┘
```

最终文件：

```text
results_analysis/comparison_prune50.mp4
```

## 代码执行过程

把整个实验按实际函数调用压缩如下：

```text
run_exp.sh
  │
  ├─ download
  │    └─ wget/unzip/tar → images + labels + 5k.txt
  │
  ├─ original
  │    └─ detect.py
  │         ├─ parse_model_cfg(yolov3.cfg)
  │         ├─ Darknet/create_modules 创建原模型
  │         ├─ load_darknet_weights 加载原权重
  │         ├─ LoadImages 逐帧预处理
  │         ├─ model(img) 前向传播
  │         ├─ non_max_suppression
  │         └─ plot_one_box/VideoWriter 保存视频
  │
  ├─ prune
  │    └─ shortcut_prune.py
  │         ├─ 加载 sparse .pt
  │         ├─ gather_bn_weights → gamma 排序/阈值/mask
  │         ├─ test.py → COCO 5k mAP
  │         ├─ 创建 compact_model 并复制保留参数
  │         └─ write_cfg + save_weights
  │
  ├─ pruned
  │    └─ detect.py（相同流程，但加载更窄的 cfg/weights）
  │
  └─ compare
       └─ ffmpeg hstack → comparison_prune50.mp4
```

`detect.py` 中每一帧的数据变化：

```text
OpenCV BGR 原视频帧
  → letterbox 保持比例缩放并填充到 416×416
  → BGR 转 RGB
  → HWC 转 CHW
  → uint8 [0,255] 转 float [0,1]
  → 增加 batch 维度，得到 [1,3,416,416]
  → YOLOv3 输出候选框
  → NMS 过滤/合并
  → 坐标映射回原视频分辨率
  → 画类别、置信度、模型名称和 FPS
  → 写入输出视频
```

`test.py` 的 mAP 数据变化：

```text
5k.txt 图片路径
  → images 替换为 labels，读取人工真实框
  → 模型预测 + NMS
  → 同类别预测框与真实框计算 IoU
  → IoU > 0.5 且未重复匹配，记为正确检测
  → 汇总不同置信度下 Precision/Recall
  → 每类 PR 曲线面积得到 AP
  → 对出现类别的 AP 求平均得到 mAP
```

## COCO mAP 和视频检测分别说明什么

`detect.py` 的视频结果是定性比较：直观看物体是否被框出、置信度是否变化、是否漏检、FPS 是否提升。但自选视频通常只覆盖少数类别，也没有人工真实框，不能严谨量化总体精度。

`test.py` 的 COCO 5k mAP 是定量比较：5000 张图片具有人工类别和边界框标注，可以统计 Precision、Recall 和 AP。剪枝脚本里运行它有三个目的：

1. 量化剪枝造成的总体精度下降；
2. 判断 50% 阈值是否破坏模型；
3. 验证“mask 置零的大模型”转换为“真实删除通道的 compact model”后精度是否一致，发现通道复制错误。

本项目默认 `test.py` 报告的是代码内部按 `iou_thres=0.5` 算出的 AP/mAP。文件名中的 `mAP53.3`、`mAP48.1` 是老师提供的模型命名信息；如果没有对应原始日志，报告中应注明“文件名标注值”，不要声称是本次实测结果。本次实测值以 `results_analysis/prune50.log` 为准。

## 结果记录和分析

实验完成后填写：

| 指标 | 原 YOLOv3 | 50% 剪枝 YOLOv3 | 变化 |
|---|---:|---:|---:|
| cfg | `yolov3.cfg` | `prune_0.5_yolov3.cfg` | 网络通道减少 |
| 参数量 | 从 `prune50.log` 抄写 | 从 `prune50.log` 抄写 | `(原-剪)/原×100%` |
| 权重文件大小 | 实测 | 实测 | `(原-剪)/原×100%` |
| COCO 5k mAP@0.5 | 实测 | 实测 | 剪枝后减原模型 |
| 输入尺寸 | 416×416 | 416×416 | 相同 |
| conf / NMS | 0.3 / 0.5 | 0.3 / 0.5 | 相同 |
| 平均视频 FPS | 从日志统计 | 从日志统计 | `剪枝/原` 倍 |
| 检测质量 | 记录目标与置信度 | 记录漏检/误检 | 定性分析 |

查看权重大小：

```bash
ls -lh weights/yolov3-full-mAP53.3.weights \
       weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights
```

从 FPS 日志提取数字并计算平均值，可以使用：

```bash
awk -F'[()]' '/FPS ==>/{sum += $2; n += 1} END{if(n) printf "frames=%d average_fps=%.3f\n", n, sum/n}' results_analysis/detect_original.log

awk -F'[()]' '/FPS ==>/{sum += $2; n += 1} END{if(n) printf "frames=%d average_fps=%.3f\n", n, sum/n}' results_analysis/detect_pruned50.log
```

结论示例结构，数值必须换成本次实测结果：

> 在相同 GPU、416×416 输入和相同检测阈值下，50% 结构化通道剪枝模型的参数量和权重文件明显下降，平均视频 FPS 提高。由于按老师要求未进行剪枝后微调，COCO 5k mAP 和部分目标置信度有所下降，远处小目标或遮挡目标更容易漏检。实验体现了结构化剪枝在模型体积、速度和检测精度之间的权衡。

## 常见问题

- **为什么不运行 `train.py`？**
  老师已提供稀疏模型，并要求剪枝后直接检测，不做耗时微调。

- **为什么下载 validation 而不下载整个 COCO？**
  本实验只用 5k 验证图片计算 mAP，不训练；约 13 GB 的 train2014 没有用途。

- **为什么数据不通过 Git 传到 GPU 设备？**
  COCO 和权重体积大，普通 GitHub 不适合存储；`5k.txt` 还包含机器相关绝对路径。GPU 设备应自己执行 `download`。

- **为什么 `git pull` 后 GPU 上没有 weights？**
  大权重已被 `.gitignore` 排除，需要通过 `scp`、网盘、共享盘或 Git LFS 单独传输。

- **出现 `CUDA out of memory` 怎么办？**
  使用 `BATCH_SIZE=8 ./run_exp.sh prune`，仍不足则改为 4。两个视频检测阶段固定 batch size 1。

- **为什么原模型与剪枝模型必须用不同 output？**
  旧版 `detect.py` 启动时会删除整个 `--output` 目录。脚本分别使用 `output_original` 和 `output_pruned50` 防止覆盖。

- **可以直接使用老师给的 60% 模型吗？**
  可以用于预先检查检测流程，但题面要求 50% 时不能将其冒充 50%。最终 50% cfg/weights 应通过 `./run_exp.sh prune` 生成。

- **Mac 能运行吗？**
  `DEVICE=cpu ./run_exp.sh original` 可以，但旧代码没有 Apple MPS；COCO 5k 验证和剪枝建议在 NVIDIA GPU 设备执行。

- **为什么视频 FPS 仍会波动？**
  每帧目标数量不同，NMS 和画框耗时不同；首批还有 CUDA 初始化和缓存开销。报告应计算多帧平均值，而不是只抄最高 FPS。
