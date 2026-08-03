- [exp4：YOLOv3 结构化通道剪枝实验](#exp4yolov3-结构化通道剪枝实验)
  - [安装运行环境](#安装运行环境)
  - [实验流程](#实验流程)
    - [下载并准备 COCO 5k 验证集](#下载并准备-coco-5k-验证集)
    - [测试原模型](#测试原模型)
    - [生成剪枝 50% 的模型](#生成剪枝-50-的模型)
    - [测试剪枝 50% 后的模型](#测试剪枝-50-后的模型)
    - [合成一个对比视频](#合成一个对比视频)
  - [其它文件/文件夹在实验里的作用](#其它文件文件夹在实验里的作用)
  - [代码执行过程](#代码执行过程)
  - [COCO mAP 和视频检测分别说明什么](#coco-map-和视频检测分别说明什么)
  - [结果记录和分析](#结果记录和分析)
  - [常见问题](#常见问题)

# exp4：YOLOv3 结构化通道剪枝实验

这个实验使用 PyTorch 版 YOLOv3，比较原始模型和结构化通道剪枝 50% 模型的检测质量、模型大小、参数量、COCO mAP 和视频处理速度。

公平比较必须满足：

- 相同输入视频；
- 相同输入尺寸，默认 `416×416`；
- 相同置信度阈值，默认 `0.3`；
- 相同 NMS IoU 阈值，默认 `0.5`；
- 相同 GPU、PyTorch 环境和精度模式；
- 视频推理都使用 batch size 1。

## 安装运行环境
 
进入 GPU 设备的仓库：
```bash
git pull
cd .../exp4
```

创建独立环境，示例使用 Python 3.10：
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与 GPU 驱动匹配的 CUDA PyTorch，再安装项目其它依赖 `python -m pip install -r requirements.txt`

检查环境：
```bash
python -c "import torch, cv2; print('torch:', torch.__version__); print('opencv:', cv2.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

输出中的 `CUDA` 应为 `True`。 
 

## 实验流程

实验流程可以按代码实际执行顺序理解：

`run_exp.sh` 是整个实验的推荐入口，负责读取用户选择的阶段、准备公共参数，并依次调用下载命令、`detect.py`、`shortcut_prune.py` 和 FFmpeg。
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
 

把下述步骤压缩成一句话就是：`run_exp.sh` 在 GPU 设备准备 COCO 5k，`detect.py` 用原 cfg/weights 生成基准视频，`shortcut_prune.py` 根据稀疏模型的 BN gamma 排序删除 50% 低重要性通道并由 `test.py` 验证 mAP，再由同一个 `detect.py` 处理同一视频，最后 FFmpeg 将两份结果左右合并；整个过程只做评估和前向传播，不运行 `train.py` 微调。


### 下载并准备 COCO 5k 验证集
当阶段是 `download` 时，`run_exp.sh` 调用 `download_coco()` 准备 COCO 5k 验证数据：
- 通过 `command -v wget` 和 `command -v unzip` 检查下载及解压工具；缺少工具时立即退出，而不是下载到一半才失败；
- `mkdir -p data/coco/images` 创建数据目录；
  - `-p` 表示目录已经存在时不报错；
- 下载并解压 COCO 2014 validation 压缩包：约 6 GB、含约 41k 张图片（本实验使用 minival JSON 从中指定 5000 张验证图片）
  - `wget -c` 下载 `val2014.zip` 到 `data/coco/images/val2014.zip`：
    - `-c` 表示断点续传，网络中断后重新运行 `download` 可以继续；
    - 不下载约 13 GB 的 `train2014`，因为老师已经提供稀疏模型，本实验不训练也不微调；
  - `unzip` 把下载来的图片解压到 `data/coco/images/val2014/`；
- 下载 Darknet 已转换好的 YOLO ground-truth 标签 `labels.tgz`，并由 `tar` 解压到 `data/coco/labels/val2014/`
  - 每张图片与标签按文件名一一对应：
    ```text
    data/coco/images/val2014/COCO_val2014_000000123456.jpg
    data/coco/labels/val2014/COCO_val2014_000000123456.txt
    ```
  - 标签每行是 `class_id x_center y_center width height`；
    - `class_id` 是 COCO 80 类中的类别编号；
    - 坐标和宽高相对于图片尺寸归一化到 `[0,1]`；
- 原来的 `5k.part` 下载地址现已失效，因此脚本改为下载 Detectron 发布的 `coco_annotations_minival.tgz`（约 74 MB），解出 `instances_minival2014.json`；其中的 `images` 数组记录标准 COCO 2014 minival 的 5000 张图片；
- 脚本使用 Python 标准库读取每条图片记录的 `file_name`，与当前机器上的 `data/coco/images/val2014/` 绝对目录拼接，生成 `data/coco/5k.txt`；
  - 使用绝对路径可以让 `test.py` 无论从哪个工作目录启动都找到图片；
  - 绝对路径和机器目录有关，所以 `5k.txt` 在 GPU 设备重新生成；
- 最后检查 `5k.txt` 是否恰好 5000 行、第一张图片是否存在、同名标签是否存在。只有三个检查都通过，download 阶段才算成功。

### 测试原模型

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
      --output results_analysis/output_original \
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
  - `opt = parser.parse_args()` 真正读取 `run_exp.sh` 传入的参数，得到 `opt.cfg`、`opt.weights`、视频 `opt.source`、阈值、设备 等属性；
    - 文件末尾的 `argparse.ArgumentParser()` 定义程序支持的所有 `--参数`
  - 接下来的都是 `with torch.no_grad()` 关闭梯度记录。目标检测只做前向传播，不需要反向传播，可节省显存和计算；
  - `detect()` 首先调用 `torch_utils.select_device(opt.device)`：
    - `--device 0` 返回 `torch.device('cuda:0')`；
    - `--device cpu` 返回 CPU；
  - 当前 `detect.py` 在开始检测时会清空 `--output` 指定的目录 (这里原模型是 `results_analysis/output_original/`)，再创建一个同名空目录，用来保证里面只保留本次运行的结果；
  - `model = Darknet(opt.cfg, img_size)` 根据 `cfg/yolov3.cfg` 创建原始网络对象。`Darknet.__init__()` 在 `models.py` 中完成：
    - `parse_model_cfg()` (在 `utils/parse_config.py` 中): 读取 cfg，把每一个 `[convolutional]`、`[shortcut]`、`[route]`、`[upsample]`、`[yolo]` 文本块转换成字典，返回保持原顺序的网络层配置字典列表
    - `create_modules()` 先创建 PyTorch 专门用于注册网络子模块的 `module_list = nn.ModuleList()`，再按 cfg 文本块从上到下循环，把每个配置字典转换成真正执行计算的 `nn.Conv2d`、`nn.BatchNorm2d`、`nn.Upsample` 或 `YOLOLayer` 等层对象，并通过 `module_list.append(modules)` 依次加入列表。
      - 因此，保存文字配置的 `module_defs[i]` 与保存实际层对象的 `module_list[i]` 下标一一对应；
      - `Darknet.forward()` 再通过 `zip(self.module_defs, self.module_list)` 同时取得每层的配置和对象，按照 cfg 的顺序完成前向传播；
  - 创建完网络后，`attempt_download(weights)` 先检查权重路径，在权重不存在时尝试取得已知公开权重（本实验的老师权重已经存在，所以不会实际下载）。随后根据文件后缀选择加载方法：
    - `.pt` 表示 PyTorch checkpoint，`torch.load(..., map_location=device)['model']` 取出其中的模型参数字典，再由 `model.load_state_dict()` 复制进网络；
    - `.weights` 表示 Darknet 二进制权重，`load_darknet_weights()` 按 cfg 层顺序依次读取并复制 BN bias、BN weight、running mean、running variance、卷积 bias 和卷积 weight；
      - 本阶段使用 `yolov3-full-mAP53.3.weights`，因此实际进入 `.weights` 分支；
  - `model.to(device).eval()` 连续完成两件事：
    - `.to(device)` 把模型参数移到 `select_device()` 返回的设备，本阶段默认是 `cuda:0`；
    - `.eval()` 切换到推理模式，使 BatchNorm 使用训练时保存的 running mean/variance，而不再根据当前视频帧更新统计量；
  - 因为 `source=data/samples/c_test.mp4` 是本地视频，不是摄像头或网络流，所以代码设置 `save_img=True`，并调用 `dataset = LoadImages(source, img_size=416, half=False)` (utils/datasets.py)：
    - `LoadImages.__init__()` 判断输入扩展名属于视频格式，把路径加入 `videos`，再由 `new_video()` 调用 `cv2.VideoCapture(path)` 打开视频并读取总帧数；
    - `for path, img, im0s, vid_cap in dataset` 每次迭代触发 `LoadImages.__next__()`:
      - 通过 `vid_cap.read()` 读取一帧原始 BGR 图像 `img0` (返回到 `detect.py` 后，这个原始帧对应变量 `im0s`)
      - `img = letterbox(img0, new_shape=416)` 保持原宽高比缩放，并补边得到模型输入尺寸；
        - `img[:, :, ::-1]` 把 OpenCV 的 BGR 通道顺序改为 RGB，`.transpose(2, 0, 1)` 再把数组从 `[H,W,C]` 改为 PyTorch 使用的 `[C,H,W]`；
        - `np.ascontiguousarray(..., dtype=np.float32)` 得到连续内存的 FP32 numpy 数组，`img /= 255.0` 把像素从 `[0,255]` 归一化到 `[0,1]`；
      - 最终返回 `path`（视频路径）、`img`（预处理帧）、`img0s`（原始帧）和 `self.cap`（视频读取对象）；
  - 为每个类别生成一种画框颜色
    - `classes = load_classes(parse_data_cfg(opt.data)['names'])`。
      - `opt.data` 是 `run_exp.sh` 传入的 `data/coco.data`（普通文本键值配置）；
      - `parse_data_cfg(opt.data)` 打开该文件，把每个 `key=value` 转成字典；
        - `['names']` 从字典取出 `data/coco.names`；
        - `classes=80`、`train` 和 `valid` 字段在本次视频检测中没有被 `detect.py` 使用；
      - `load_classes()` 按行读取这个 names 文件 `coco.names`，生成类别名称列表 `classes` (“类别编号 → person、car 等名称”)
    - `colors` 每个类别随机生成的一种画框颜色，同一次程序运行中相同类别使用相同颜色；
  - 进入逐帧循环后，每一帧按以下顺序处理：
    - 如果使用 CUDA，先调用一次 `torch.cuda.synchronize()`，等待上一帧 GPU 任务结束，然后用 `t = time.time()` 开始本帧计时
    
    
    
    
    
    
    
    - `torch.from_numpy(img).to(device)` 把 `[3,416,416]` numpy 数组转换成 tensor 并复制到 GPU；
      - 当 `img.ndimension() == 3` 时，`unsqueeze(0)` 在最前面加入 batch 维，得到 `[1,3,416,416]`；
      - `pred, _ = model(img)` 执行前向传播，合并三个 YOLO 检测尺度产生的候选框。这里的候选框还没有经过置信度过滤和去重；
    - `non_max_suppression(pred, 0.3, 0.5)` 先过滤 object confidence 低于 `0.3`、宽高过小或包含非有限数值的候选框，再按类别处理重叠框；当前代码使用 `MERGE` 模式，对 IoU 大于 `0.5` 的同类别框按置信度加权合并；
    - 对 NMS 返回的每张图检测结果 `det`，本地视频分支令 `p=path`、`im0=im0s`，并用 `save_path = output_original/c_test.mp4` 确定输出路径；
    - `scale_coords(img.shape[2:], det[:, :4], im0.shape)` 把基于 416 输入的 `xyxy` 框坐标映射回原始视频分辨率，再调用 `.round()` 取整；
    - 按类别统计当前帧检测数量；随后遍历每个框，从 `det` 中取出 `xyxy`、object confidence 和 class id，形成例如 `person 0.93` 的标签，再由 `plot_one_box()` 画到原始帧 `im0`；
    - 画框完成后再次调用 `torch.cuda.synchronize()`，用 `frame_seconds = time.time() - t` 和 `frame_fps = 1 / frame_seconds` 计算当前帧 FPS；这个计时包含 tensor 转换、GPU 前向传播、NMS、坐标恢复和画框，但不包含计时前的视频读取/预处理，也不包含后面的文字叠加与视频写盘；
    - `cv2.rectangle()` 先在左上角画黑色背景，`cv2.putText()` 再写入 `Original YOLOv3 | FPS ...`；`print()` 同时把 FPS 输出到终端，并由 `tee` 保存到 `detect_original.log`；
    - 第一次写视频时，`cv2.VideoWriter()` 从 `vid_cap` 取得原视频 FPS、宽和高，并使用 `mp4v` 编码创建 `results_analysis/output_original/c_test.mp4`；之后每帧通过 `vid_writer.write(im0)` 追加到同一个输出视频；
  - 所有帧处理完成后，程序打印结果目录和 `Done. (...s)` 总耗时；在 GPU Linux 设备上不会执行仅针对 macOS 的自动打开文件分支。

### 生成剪枝 50% 的模型

当阶段是 `prune` 时，`run_exp.sh` 调用 `prune_model()`：
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

老师提供的 `sparse-yolov3-full-mAP48.1.pt` 是“稀疏训练”的结果，本实验直接从这个文件开始，不实际运行 `train.py`。它从普通 YOLOv3 到可剪枝稀疏模型的生成原理如下：

- `train.py -sr --prune 1` 根据 `yolov3.cfg` 创建通道尚未减少的完整 YOLOv3，并加载已经完成普通训练的初始权重；
- `parse_data_cfg()` 从 `coco.data` 的 `train` 字段取得训练图片列表，`LoadImagesAndLabels` 和 `DataLoader` 完成图片增强、标签读取和 batch 组合；
- `parse_module_defs2()` 找出允许稀疏化的卷积 + BN 层，并记录 shortcut 两侧的通道对应关系：
  - shortcut 执行逐元素相加，两侧通道必须对应，因此不是每个 BN 层都能互相独立地删通道；
- 每个 batch 先执行完整模型的正常训练：
  - `model(imgs)` 前向传播；
  - `compute_loss()` 计算边界框回归、objectness 和分类损失；
  - `loss.backward()` 计算检测任务对所有模型参数的梯度；
- 开启 `-sr/--sparsity-regularization` 后，`BNOptimizer.updateBN()` 在反向传播得到正常梯度之后，对每个可剪枝 BN 的 gamma 梯度额外加入：
  ```text
  s × sign(gamma)
  ```
  - 这等价于对 gamma 加 L1 稀疏正则；gamma 为正时推动它减小，为负时推动它增大，使其绝对值逐渐靠近 0；
  - 默认 `s=0.001`，训练进行到后一半时降为原来的 1%，避免后期持续过强地压缩 gamma；
- `optimizer.step()` 同时应用正常检测梯度和 gamma 稀疏梯度，因此模型在保持检测能力的同时，逐渐把不重要通道的 gamma 压到接近 0；
- 每个 epoch 后，`test.py` 在验证集计算 mAP，训练代码依据 mAP 更新 `best_fitness`；
- `torch.save()` 保存 `weights/last.pt`、`weights/best.pt` 和定期备份，其中 `model` 字段包含完整网络参数：
  - 此时只是 gamma 呈稀疏分布，cfg、卷积张量尺寸、通道数和参数量都还没有减少；
  - 老师从该流程得到并提供了 `sparse-yolov3-full-mAP48.1.pt`，所以本实验跳过上述耗时训练，直接把它交给 `shortcut_prune.py`。

上述稀疏训练只负责让通道“容易被区分重要性”。接下来的结构化剪枝才真正删除通道：

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
  - 再次读取 `data/coco.data`，这次使用 `valid=data/coco/5k.txt` 找到 5000 张验证图片；
  - `LoadImagesAndLabels` 读取图片列表，并将路径中的 `images` 替换为 `labels`，寻找同名真实标注；
  - 数据加载器此时第一次读取 `data/5k.shapes`：
    - 它不是 Python 文件，而是 5000 张验证图片原始尺寸的文本缓存，每行两个数字表示一张图片的 `宽 高`；
    - 缓存存在且行数正确时直接使用，以免重新打开全部图片读取尺寸；缓存不存在或不匹配时会扫描图片并自动重新生成；
  - `DataLoader` 按 `--batch-size 16` 组合 batch；显存不足时可以通过 `BATCH_SIZE=8` 降低；
  - 每个 batch 执行模型前向传播和 NMS；
  - 对每张图片：
    - 取预测类别、置信度和预测框；
    - 取真实类别和真实框；
    - 只有预测类别正确、预测框和真实框 IoU 大于 `0.5`，并且该真实框尚未被另一个预测匹配，才把该预测标为正确；
  - `ap_per_class()` 汇总不同置信度位置的 Precision 和 Recall，计算每个类别 PR 曲线下的面积 AP；
  - 对参与评估类别的 AP 取平均得到 mAP；
  - `test.py` 只评估模型，不执行 `loss.backward()` 和 `optimizer.step()`，所以 COCO 验证不会修改模型参数。

### 测试剪枝 50% 后的模型

当阶段是 `pruned` 时，`run_exp.sh` 调用 `test_pruned()`：
- 使用的 Python 仍然是同一个 `detect.py`，数据读取、前向传播、NMS、画框和视频保存流程也完全相同；
- `detect.py` 会再次读取 `data/coco.data` 中的 `names` 字段，用于显示检测类别名称；
- 唯一关键变化是模型配对改成：
    ```text
    cfg/prune_0.5_yolov3.cfg
    weights/prune_0.5_sparse-yolov3-full-mAP48.1.weights
    ```
- 输入仍为同一 `VIDEO`，尺寸仍为 416，置信度/NMS 阈值仍为 0.3/0.5，设备仍为同一 GPU；
- 视频左上角文字改为 `Pruned 50% YOLOv3 | FPS ...`；
- 结果保存到 `results_analysis/output_pruned50/`，日志保存到 `results_analysis/detect_pruned50.log`；
- 两边复用完全相同的推理代码，因此观察到的主要差异来自网络结构和模型权重，而不是两套实现不同。

### 合成一个对比视频

当阶段是 `compare` 时，`run_exp.sh` 调用 `compare_videos()`：
- `basename "$VIDEO"` 只取得输入文件名，例如 `my_video.mp4`；
- 分别检查 `results_analysis/output_original/my_video.mp4` 和 `results_analysis/output_pruned50/my_video.mp4`；
- FFmpeg 的两个 `scale=-2:540` 把视频缩放到相同高度 540，并让宽度自动取合法偶数；
- `hstack=inputs=2` 按相同时间顺序把原模型放左边、剪枝模型放右边；
- `libx264 + yuv420p` 生成兼容性较好的 H.264 MP4；
- 最终保存为 `results_analysis/comparison_prune50.mp4`。
- `run_exp.sh` 全部完成后，实验的主要结果对应关系为：
  - `results_analysis/prune50.log`：回答剪枝前后 mAP、参数量和模型前向时间变化；
  - `results_analysis/detect_original.log`：原模型逐帧 FPS；
  - `results_analysis/detect_pruned50.log`：剪枝模型逐帧 FPS；
  - `results_analysis/output_original/<视频名>`：原模型可视化检测结果；
  - `results_analysis/output_pruned50/<视频名>`：剪枝模型可视化检测结果；
  - `results_analysis/comparison_prune50.mp4`：最终提交的左右对比视频。
 

## 其它文件/文件夹在实验里的作用
- `requirements.txt`：记录除 PyTorch 外的 Python 依赖。
- `run_exp.sh`：整个实验的入口，按所选阶段调用数据下载、原模型检测、模型剪枝、剪枝模型检测和视频合并流程。
- `detect.py`：执行图片或视频推理，并绘制检测框、类别、模型名称和 FPS。
- `shortcut_prune.py`：根据稀疏模型的 BN gamma 生成 50% 结构化通道剪枝模型。
- `test.py`：使用 COCO 5k 的真实标签计算 Precision、Recall、mAP 和 F1。
- `train.py`：提供训练、稀疏训练和微调功能；本实验不执行。
- `models.py`：解析 cfg，创建 Darknet/YOLOv3 网络，并负责读取和保存模型权重。
- `cfg/`：保存模型网络结构。
  - `yolov3.cfg`：原始 YOLOv3 的网络结构。
  - `prune_0.5_yolov3.cfg`：执行剪枝后生成的 50% 剪枝网络结构，不需要手工编写。
- `weights/`：保存实验输入权重和生成的剪枝权重。
  - `yolov3-full-mAP53.3.weights`：原始 YOLOv3 权重。
  - `sparse-yolov3-full-mAP48.1.pt`：稀疏训练模型，剪枝阶段的输入。
    - 大权重已被 .gitignore 排除，所以 `$ git pull` 后 GPU 上没有 weights
  - `prune_0.5_sparse-yolov3-full-mAP48.1.weights`：执行剪枝后生成的 50% 剪枝权重。
- `data/`：保存 COCO 配置、类别名称、验证数据和测试视频；其中各配置文件和缓存的读取过程在实验流程首次用到时说明。
  - `data/samples/c_test.mp4`：老师提供的示例输入视频。
  - `data/coco/`：由 `download` 阶段生成，保存 COCO 5k 验证图片、YOLO 格式标签和图片路径列表。
- `utils/`：提供数据加载、NMS、AP 计算、坐标处理和剪枝辅助函数。
- `results_analysis/`：保存两个模型的检测视频、运行日志和最终左右对比视频。
  - `output_original/`、`output_pruned50/`：分别保存原模型和 50% 剪枝模型的检测视频。
  - `prune50.log`：保存剪枝过程以及 COCO mAP、参数量和前向时间结果。
  - `detect_original.log`、`detect_pruned50.log`：分别保存两个模型的视频检测日志和逐帧 FPS。
  - `comparison_prune50.mp4`：原模型与剪枝模型的最终左右对比视频。

三个主要模型的关系：

```text
原始模型
yolov3.cfg + yolov3-full-mAP53.3.weights
                 │
                 │ 完成稀疏训练
                 ▼
稀疏模型
yolov3.cfg + sparse-yolov3-full-mAP48.1.pt
                 │
                 │ shortcut_prune.py --percent 0.5
                 ▼
50% 结构化剪枝模型
prune_0.5_yolov3.cfg + prune_0.5_sparse-yolov3-full-mAP48.1.weights
```
- `.cfg` 描述网络每层结构，`.weights`/`.pt` 保存参数值。
  - 原 cfg 和剪枝权重不能混用，因为剪枝后各层通道数已经改变。
- `--percent 0.5` 表示在全部可剪枝 BN gamma 的排序中以 50% 位置确定全局阈值，不表示最终参数量或模型文件一定刚好减少 50%。shortcut 层需要保持相加两侧通道一致，也会影响实际剪枝率。


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
完整实验输出包括：

1. 原 YOLOv3 对同一视频的检测结果；
2. 从稀疏模型实际生成的 50% 通道剪枝 cfg 和 weights；
3. 剪枝模型对同一视频的检测结果；
4. 原模型和剪枝模型的 COCO 5k mAP、参数量和推理时间；
5. 左侧原模型、右侧剪枝模型的合并视频；
6. 对速度提升、文件缩小、置信度变化、误检和漏检的分析。


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
- 出现 `CUDA out of memory` 怎么办？
  使用 `BATCH_SIZE=8 ./run_exp.sh prune`，仍不足则改为 4。两个视频检测阶段固定 batch size 1。

- **为什么原模型与剪枝模型必须用不同 output？**
  当前 `detect.py` 每次启动时都会清空 `--output` 指定的目录，以免上一次运行留下的图片或视频混入本次结果。脚本分别使用 `results_analysis/output_original` 和 `results_analysis/output_pruned50`，防止剪枝模型运行时清除原模型结果。
 
- **为什么视频 FPS 仍会波动？**
  每帧目标数量不同，NMS 和画框耗时不同；首批还有 CUDA 初始化和缓存开销。报告应计算多帧平均值，而不是只抄最高 FPS。
