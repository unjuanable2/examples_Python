- [exp3：TensorRT / INT8 推理实验](#exp3tensorrt--int8-推理实验)
  - [整体过程](#整体过程)
    - [其它文件/文件夹在实验里的作用](#其它文件文件夹在实验里的作用)
  - [推理结果](#推理结果)
  - [Questions](#questions)


# exp3：TensorRT / INT8 推理实验

## 整体过程
（模型部署 / 推理加速入门）

这个实验的主任务不是重新训练模型，而是把已经训练好的 PyTorch 模型拿来做推理加速。大致目标是：先用普通 PyTorch 模型推理，再用 `torch2trt` 把模型转换成 TensorRT 模型，然后比较两种推理方式在同一张图片上的耗时和预测结果。

推理流程可以按代码执行顺序理解：

- `int8_infer.py` 是推理入口，负责解析命令行参数。
  ```bash
  python int8_infer.py --model alexnet --gpu
  # --model alexnet：指定要加载 alexnet 这个模型结构和权重；
  # --gpu：表示必须使用 CUDA GPU。TensorRT 推理依赖 NVIDIA GPU，所以这里必须开 GPU。
  ```
- 解析完命令行参数后，`int8_infer.py` 通过 `model_factory(model_name)` 创建模型对象。
  - 这里的 `model_factory` 来自 `models`。
  - 也就是说，`exp3` 的推理脚本需要能找到 `exp1+2/models/` 里的模型定义。
  - 当命令行参数是 `--model alexnet` 时，代码会创建 AlexNet 结构的模型对象。
- 如果传入了 `--gpu`，并且当前机器能使用 CUDA，代码会做几件事：
  - `torch.backends.cudnn.enabled = True`：启用 CuDNN，让 PyTorch 可以调用 NVIDIA 对卷积等操作做过优化的底层库；
  - `torch.backends.cudnn.benchmark = True`：让 CuDNN 根据当前输入尺寸自动尝试并选择更快的卷积算法；
  - `model = model.cuda()`：把模型从 CPU 内存移动到 GPU 显存上。
  - 如果没有传 `--gpu`，或者机器不能使用 CUDA，程序会直接退出。
- `model.load_state_dict(torch.load('./weights/'+model_name+'.pt')['net'])` 加载已经训练好的模型权重。
  - `model_factory(model_name)` 只是创建“空模型结构”；
  - `weights/alexnet.pt` 里保存的是训练好的参数；
  - 只有把权重加载进去，这个模型才是已经训练好的模型。
- `dataset.py` 负责准备两类数据：
  - `testloader`：从 CIFAR-10 测试集读取图片，用来测试模型整体准确率；
  - `QDataset`：从 `data/q/*.jpg` 读取图片，原本是给 INT8 量化校准使用的。
    - INT8 量化需要一批代表性图片，让 TensorRT 估计模型中间结果的大致数值范围；
    - 当前 `int8_infer.py` 里创建了 `cali_cifar10 = QDataset(...)`，但真正传给 `torch2trt()` 的校准参数被注释掉了，所以现在这份代码更像是“准备了 INT8 校准数据，但校准调用还没有完整启用”。
- `model.eval()` 把模型切换到评估模式。
  - 推理时不需要 Dropout 的随机行为；
  - BatchNorm 也应该使用训练后保存下来的统计量，而不是继续按当前 batch 更新统计量。
- `x = torch.randn([1, 3, 32, 32]).cuda()` 创建一个假的输入张量，形状是 `[1, 3, 32, 32]`。
  - `1` 表示 batch size 是 1；
  - `3` 表示 RGB 三个通道；
  - `32 x 32` 对应 CIFAR-10 图片大小；
  - `torch2trt()` 需要一个示例输入，才能知道模型输入长什么样，并据此转换 TensorRT engine。
- `model_trt_int8 = torch2trt(model, [x], fp16_mode=True, int8_mode=True)` 尝试把 PyTorch 模型转换成 TensorRT 模型。
  - `model` 是原始 PyTorch 模型；
  - `[x]` 是示例输入；
  - `fp16_mode=True` 表示允许 TensorRT 使用 FP16 半精度计算；
  - `int8_mode=True` 表示尝试使用 INT8 推理；
  - 但下面这些更完整的 INT8 校准参数目前被注释掉了：
    ```python
    # max_batch_size=1,
    # int8_calib_dataset=cali_cifar10,
    # int8_calib_algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    ```
  - 所以如果以后要认真做 INT8 量化，需要重点检查这几行是否应该恢复。
- 后面的大段 `test(model)` 和 `test(model_trt_int8)` 被三引号注释掉了。
  - 如果恢复这段代码，它会遍历 CIFAR-10 测试集，比较 PyTorch 模型和 TensorRT 模型在整个测试集上的准确率和耗时；
  - 当前实际执行的是下面的单张图片推理流程。
- `test.jpg` 是单张测试图片。
  - 代码用 `Image.open(test_image)` 打开图片；
  - 再用 `cali_augmentation` 做 Resize、ToTensor、Normalize；
  - `img_tensor.unsqueeze_(0)` 在最前面加一个 batch 维度，把图片从 `[3, 32, 32]` 变成 `[1, 3, 32, 32]`；
  - 最后 `.cuda()` 把图片数据移动到 GPU。
- 对同一张图片分别做两次推理：
  - `y_fp32 = model(img_tensor)`：用原始 PyTorch 模型推理；
  - `y_int8 = model_trt_int8(img_tensor)`：用 TensorRT 模型推理；
  - 每次推理后调用 `torch.cuda.synchronize()`，是为了等待 GPU 真正执行完再计时。
    - CUDA 操作默认是异步的；
    - 如果不 `synchronize()`，CPU 可能只是把任务提交给 GPU 就继续往下走，计时会偏小。
- `torch.softmax(y_fp32[0], dim=0) * 100` 把模型输出的 logits 转成每个类别的概率百分比。
  - `torch.max(...)` 找出概率最大的类别；
  - `classes = ['plane','car','bird','cat','deer','dog','frog','horse', 'ship','truck']` 是 CIFAR-10 的 10 个类别名。
- 最后代码把两行结果画到 `test.jpg` 上：
  - 第一行：PyTorch FP32 模型的推理耗时、预测类别、置信度；
  - 第二行：TensorRT INT8 模型的推理耗时、预测类别、置信度；
  - 然后 `img.save(test_image, 'jpeg')` 会把结果覆盖保存回 `test.jpg`。


把这些步骤压缩成一句话就是：`int8_infer.py` 读取参数，创建模型结构，加载 `weights/` 里的训练权重，使用 `torch2trt` 把 PyTorch 模型转换成 TensorRT 模型，再用同一张 `test.jpg` 分别跑 PyTorch 推理和 TensorRT 推理，比较推理时间和预测类别。


### 其它文件/文件夹在实验里的作用
- `int8_infer.py`：主推理脚本，负责加载模型、转换 TensorRT、计时、预测、把结果画到图片上。
- `dataset.py`：数据集准备脚本。
  - `testloader` 用 CIFAR-10 测试集做模型准确率测试；
  - `QDataset` 从 `data/q/*.jpg` 读取图片，原本用于 INT8 校准数据。
- `data/q/`：量化校准图片。文件名里带类别数字，`QDataset.__getitem__()` 会从文件名中解析标签。
- `test.jpg`：单张推理测试图片。程序运行后会把 FP32 / INT8 的预测结果和耗时写到这张图片上。
- `weights/alexnet.pt`：已经训练好的 AlexNet 权重文件。没有这个权重文件，`int8_infer.py` 只能创建模型结构，不能得到训练后的模型参数。

## 推理结果
terminal 输出一般会包含类似：

```text
Testing model: alexnet
Time Spent for fp32: ...ms
Time Spent for int8: ...ms
Mode: fp32, ...ms <class> (...%)
Mode: int8, ...ms <class> (...%)
```

同时，`test.jpg` 会被覆盖保存，上面会写出 FP32 和 INT8 两种模式的预测类别、置信度和推理耗时。

## Questions
`int8_infer.py`
??? 现在 `torch2trt()` 里虽然写了 `int8_mode=True`，但校准数据集参数被注释掉了；后续需要确认这是否是真正完整的 INT8 量化流程。

`dataset.py`
??? `root='/opt/DATASET/CIFAR/'` 是老师机器上的数据集路径。如果自己的电脑没有这个目录，需要改成自己的 CIFAR-10 路径，或者改成 `root='./data'` 让代码下载/读取当前实验目录下的数据。
