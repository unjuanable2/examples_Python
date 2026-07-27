- [exp3：AlexNet TensorRT INT8 推理实验](#exp3alexnet-tensorrt-int8-推理实验)
  - [整体过程](#整体过程)
    - [其它文件/文件夹在实验里的作用](#其它文件文件夹在实验里的作用)
    - [INT8 量化和校准的原理补充](#int8-量化和校准的原理补充)
  - [结果 和 分析](#结果-和-分析)


# exp3：AlexNet TensorRT INT8 推理实验
（模型部署 / INT8 推理加速入门）

这个实验使用 exp2 训练好的 FP32 AlexNet，通过 torch2trt 构建 TensorRT INT8 engine，并比较 PyTorch FP32 与 TensorRT INT8 的单图预测、平均推理延迟以及完整 CIFAR-10 测试集准确率。

## 整体过程

- 运行前需要把 exp2 训练得到的 FP32 checkpoint 放到：
  ```text
  exp3/weights/alexnet/weights.130.83.050.pt
  ```
  - 文件名遵循 exp2 的保存格式 `weights.<epoch>.<accuracy>.pt`；
  - checkpoint 必须包含 `net` 模型参数和 `acc` 准确率；
  - 权重文件被根目录 `.gitignore` 中的 `*.pt` 忽略，不会提交到 Git；
  - 找不到权重时，程序会在 TensorRT 转换前直接报告明确的 `FileNotFoundError`。
- 推荐在项目根目录执行：
  ```bash
  ./exp3/run_exp.sh
  ```
  也可以进入 exp3 后执行 `./run_exp.sh`。
- `run_exp.sh` 会：
  - 根据脚本自身位置进入 exp3，因此相对路径不会受启动目录影响；
  - 创建 `results_analysis/`；
  - 把终端标准输出和错误同时显示并保存到 `results_analysis/run_exp3_out.txt`；
  - 执行 `python3 int8_infer.py --gpu --model alexnet`。
- `int8_infer.py` 把 exp2 放到 Python 模块搜索路径最前面，因此：
  ```python
  from models import model_factory
  ```
  明确导入 `exp2/models`。`model_factory("alexnet")` 创建 AlexNet 结构，随后程序加载 `exp3/weights/alexnet/weights.130.83.050.pt` 中的训练参数并切换到 `eval()` 模式。
- `data.py` 准备两种数据：
  - `testloader` 从 `exp1/data/` 读取完整 CIFAR-10 测试集，`batch_size=1`；
  - `QDataset` 从 `exp3/data/q/*.jpg` 读取 INT8 校准图片；每个样本经过 Resize、ToTensor 和 Normalize 后变成 `[1, 3, 32, 32]` CUDA 张量，并按 torch2trt 要求返回 `[input_tensor]`，不返回 label。
- `int8_infer.py` 使用与 exp2 一致的预处理：
  - Resize 到 `32×32`；
  - ToTensor；
  - 使用 CIFAR-10 的 RGB 均值和标准差 Normalize。
- TensorRT INT8 engine 通过下面的配置构建：
  ```python
  model_trt_int8 = torch2trt(
      model,
      [x],
      fp16_mode=False,
      int8_mode=True,
      max_batch_size=1,
      int8_calib_dataset=cali_cifar10,
      int8_calib_algorithm=tensorrt.CalibrationAlgoType.ENTROPY_CALIBRATION_2,
  )
  ```
  - `fp16_mode=False`：不主动启用 FP16；
  - `int8_mode=True`：为支持的层启用 INT8；
  - `int8_calib_dataset`：使用代表性图片校准 activation values；
  - `ENTROPY_CALIBRATION_2`：使用熵校准算法选择 INT8 动态范围。
- 单图测试读取干净的 `test.jpeg`，FP32 和 INT8 使用同一个已经放到 GPU 的 `img_tensor`：
  - 两种模型各预热 50 次，预热不计入结果；
  - 两种模型各正式推理 100 次；
  - 使用 `time.perf_counter()` 计时，并在计时前后正确同步 CUDA；
  - 总时间除以 100，得到平均单次 latency；
  - 最后一次输出用于计算 softmax、预测类别和 confidence。
- 程序把两种模型的平均 latency、prediction 和 confidence 写到 `test_result.jpg`，不会覆盖干净输入 `test.jpeg`。
- 最后，程序让 FP32 和 INT8 分别遍历完整 CIFAR-10 测试集，输出 accuracy 和端到端总耗时。该总耗时包含 DataLoader、预处理、CPU 到 GPU 数据复制、Python 循环和模型推理。

把流程压缩成一句话就是：`run_exp.sh` 保存完整日志，`data.py` 提供测试集和校准集，`int8_infer.py` 从 exp2 创建 AlexNet、加载 exp3 中的训练权重、构建 INT8 engine，再比较 FP32 和 INT8 的单图平均 latency、预测结果与完整测试集 accuracy。


### 其它文件/文件夹在实验里的作用

- `run_exp.sh`：一键运行入口，并保存完整终端日志。
- `int8_infer.py`：模型加载、INT8 校准与转换、单图测试、结果图片和完整测试集评估。
- `data.py`：读取 `exp1/data/` 中的 CIFAR-10 测试集，并定义 `data/q/` 校准集。
- `data/q/`：INT8 calibration images，不用于训练模型。
- `weights/.gitkeep`：让 Git 保留 weights 空目录；实际 `.pt` 权重被忽略。
- `weights/alexnet/weights.130.83.050.pt`：本实验需要的 FP32 AlexNet checkpoint，需在运行机器上自行放入。
- `test.jpeg`：干净单图输入，不会被程序覆盖。
- `test_result.jpg`：运行后生成的单图标注结果。
- `results_analysis/run_exp3_out.txt`：一键脚本保存的完整 terminal 输出。
- `../exp2/models/`：AlexNet 模型结构来源。

### INT8 量化和校准的原理补充

INT8 PTQ 通常同时涉及 weights 和 activation values。设置 `int8_mode=True` 后，TensorRT builder 会读取训练好的 FP32 weights，为支持 INT8 的层计算权重量化 scale；权重数值本身已经确定，因此通常不需要用校准图片估计其范围。

Activation values 会随输入图片变化，所以需要校准集。TensorRT 让 `data/q/` 图片经过网络，统计各层 activation values 的分布。`ENTROPY_CALIBRATION_2` 的过程可以概括为：

1. 建立 activation values 的分布/直方图；
2. 尝试不同截断阈值 $T$；
3. 模拟截断与 INT8 量化后的分布；
4. 计算量化前后分布的 KLD（Kullback-Leibler Divergence）；
5. 选择 KLD 最小的 $T$，再由 $T$ 推导 INT8 scale。

以常见的对称有符号 INT8 为例，可以近似理解为：

$$scale=T/127, \qquad q=\operatorname{round}(x/scale)$$

校准图片与正式图片必须使用相同预处理，否则校准阶段观察到的 activation 分布可能不适合正式输入。还要注意，没有 INT8 实现的算子可能回退到其它精度，因此启用 INT8 不代表 engine 中每个算子一定执行 INT8。

## 结果 和 分析

一键运行后会生成：

```text
exp3/
├── test_result.jpg
└── results_analysis/
    └── run_exp3_out.txt
```

终端和日志会包含：

- FP32 的 100 次平均单图 latency；
- TensorRT INT8 的 100 次平均单图 latency；
- 两种模型对 `test.jpeg` 的 prediction 和 confidence；
- 两种模型在完整 CIFAR-10 测试集上的 accuracy；
- 两种模型遍历完整测试集的端到端总耗时。

单图平均 latency 主要测已经位于 GPU 上的同一个张量重复执行模型的时间。完整测试集总耗时还包含数据读取、预处理和数据传输，因此这两种指标不能交叉比较。正确的比较方式是：FP32 单图平均 latency 对比 INT8 单图平均 latency；FP32 完整测试集总时间对比 INT8 完整测试集总时间；FP32 accuracy 对比 INT8 accuracy。
