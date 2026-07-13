- [训练模型的整体过程](#训练模型的整体过程)
  - [其它文件/文件夹在实验里的作用](#其它文件文件夹在实验里的作用)
- [Questions](#questions)


# 训练模型的整体过程
（Pytorch使用入门）

这个实验的主任务是用 PyTorch 在 CIFAR-10 上训练一个图像分类模型。CIFAR-10 每张图像是 `32x32` 的 RGB 图片，标签一共有 10 类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。

训练流程可以按代码执行顺序理解：

- `main.py` 是训练入口，负责解析命令行参数。
  ```bash
  python main.py --model resnet18 --steps 200 --lr 0.2 --gpu
  # --model resnet18：指定使用模型 resnet18（已写好）
  # --steps 200：训练 200 个 epoch
  # --lr 0.2：初始学习率为 0.2
  # --gpu：如果机器有 CUDA，就把模型和数据放到 GPU 上训练；
  # `--fp16`、`--loss_scaling`：可选的半精度训练相关参数。
  ```
- 解析完命令行参数后，`main.py` 传参数(model name)，通过 `models/model_factory_dict.py` 创建模型。
  - `models/` 文件夹里保存了多种网络结构定义，e.g. `resnet.py`、 `alexnet.py`、 `vgg.py`、 `mobilenet.py` 等；
    - `models/__init__.py` 把 `model_factory` 暴露出来；
    - `models/model_factory_dict.py` 维护“模型名称字符串 -> 模型构造函数”的映射；
  - 当命令行参数是 `--model resnet18` 时，实际会调用 `models/resnet.py` 里的 `ResNet18()`: 构造的是 `ResNet(BasicBlock, [2, 2, 2, 2])`，最后的全连接层输出 10 个值，对应 CIFAR-10 的 10 个类别。  
- `main.py` 把创建出的模型（模型名称、模型对象、模型学习率、学习率、是否使用 GPU、是否使用 FP16、是否使用 loss scaling）交给 `train.py` 里的 `Trainer` 类，得到 `Trainer` 类训练器对象 `trainer`。
  - `Trainer.__init__()` 保存模型、学习率、是否使用 GPU/FP16 等配置；
    - 初始化 对象属性 为 用户传进来的参数
    - 如果启用 GPU，会执行 `model.cuda()` 把模型移动到 GPU 显存上
    - 优化器使用 `optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)`；
      - `momentum=0.9` 可以让参数更新带有“惯性”，通常收敛更稳；
      - `weight_decay=5e-4` 是 L2 正则化，用来抑制权重过大，减少过拟合；
    - 学习率调度器是 `MultiStepLR`，到 `[10, 20, 50, 100, 180]` 这些 epoch 时把学习率乘以 0.1
- `main.py` 导入 `data.py` 中的两个数据加载器对象 `trainloader`、`testloader`，完成数据准备。`data.py` 具体涉及：
  - 定义训练集和测试集的预处理流程：
    - 训练集使用 `train_transforms` (一种训练预处理流程)：
      - `RandomCrop(32, padding=4)`：先在原图四周补 4 像素，再随机裁回 `32x32`。这会让同一张图在不同 epoch 里稍微平移，模型不能只记住物体固定出现在某个像素位置，从而提升泛化能力。
        - 如果去掉它，训练集精度可能仍然很高，但测试集更容易下降。
      - `RandomHorizontalFlip()`：以 50% 概率水平翻转图片。CIFAR-10 里的车、飞机、猫、狗等左右翻转后类别通常不变，所以这相当于扩充训练数据。
        - 如果任务是识别文字、交通标志方向等，随便翻转就可能破坏标签。
      - `ToTensor()`：把 PIL/numpy 图像变成 PyTorch 张量，并把像素值从 `[0, 255]` 缩放到 `[0, 1]`。
        - 归一化后再做图像增强不方便，也容易破坏已经标准化好的数值分布，所以前两步先做图像增强。
      - `Normalize(mean, std)`：按 CIFAR-10 的 RGB 均值和标准差做标准化，让输入分布更稳定，通常能让梯度下降更容易收敛。
    - 测试集使用 `test_transforms` (一种训练预处理流程)：只做 `ToTensor()` 和 `Normalize()`；
      - 不做随机裁剪和随机翻转，因为测试时要稳定评估同一批图像，不能每次评估都随机改变输入。
  - `trainset` (训练集)、`testset` (测试集) 是 `torchvision.datasets.CIFAR10` 数据集对象
    - 每个训练集每次取出训练图片时，应用上面定义的训练预处理流程。
  - `trainloader`、`testloader` 是 `torch.utils.data.DataLoader` 对象，作用是把数据集切成一批一批的小 batch
    - `batch_size=128` 表示每次训练读 128 张图；
    - `shuffle=True` 表示每个 epoch 打乱样本顺序，避免模型总是按固定顺序见到数据；
    - `num_workers=4` 表示用 4 个子进程并行读取数据，加快喂数据速度。
- `main.py` 最后一行 `trainer.train_and_evaluate(trainloader, testloader, args.steps)` 开始训练和评估循环。外层循环跑 `steps` 次，也就是 epoch 数。每个 epoch:
  - 先调用 `train()` 在训练集上更新参数
    - `self.model.train()` 切换到训练模式，e.g. 启用 BatchNorm/Dropout 的训练行为
    - 设置 `train_loss` (累计训练 loss), `correct` (累计预测正确的样本数), `total` (累计已经处理过的样本数) 为 0
    - 设置当前 epoch 的学习率：
      - 前 5 个 epoch (从0开始计数) 使用 warmup，让学习率从较小值逐步升到设定的初始学习率，避免一开始步子太大导致训练不稳定；
      - 第 5 个 epoch：恢复到初始学习率；
      - 第 5 个 epoch 之后：每个 epoch 末尾调用 `self.scheduler.step();` :当 scheduler 走到 milestones=[10, 20, 50, 100, 180] 时，学习率会乘以 gamma=0.1
    - 定义损失函数 `nn.CrossEntropyLoss()`。
      - 分类任务通常不用 `MSELoss()`，因为模型输出的是每一类的 logits，交叉熵更适合“10 类里选 1 类”的监督学习；
    - 遍历训练集的每一个 batch，每个 batch 的训练顺序是：
      - 从 `trainloader` 取出 `inputs` 和 `targets`；
        - 如果使用 GPU，把数据移到 CUDA；
        - 清空上一轮 batch 留下的梯度
      - `outputs = model(inputs)` 前向传播，得到 10 类 logits；
      - `loss = criterion(outputs, targets)` 根据模型预测结果 outputs vs 真实训练标签 targets，计算分类损失；
      - `loss.backward()` 反向传播，根据 loss 计算模型每个参数的梯度
      - `optimizer.step()` 根据梯度更新模型参数；
      - 统计训练 loss `train_loss` 和 accuracy ( 根据 `total`, `correct`) ，并用 `utils.py` 的 `progress_bar()` 打印进度条。
        - 关于 `utils.py`：详情见注释
          - `progress_bar()`: 这个函数用到了 `format_time(seconds)`
          - `format_time(seconds)`: 耗时格式化

---

  - 再调用 `evaluate()` 在测试集上计算准确率；
    - `self.model.eval()` 切换到评估模式；
    - `with torch.no_grad()` 关闭梯度记录，节省显存和计算；
    - 遍历 `testloader`，只做前向传播和准确率统计，不更新参数；
    - `outputs.max(1)` 取 logits 最大的类别作为预测类别；
    - 如果当前测试准确率超过历史最好值， `self.best_acc`，就调用 `save_model()` 保存权重。
      - `weights/` 是训练结果保存位置。
        - 普通训练会保存到 `weights/<model_name>/weights.<epoch>.<acc>.pt`；
        - FP16 训练会保存到 `weights/<model_name>_fp16/`；
        - `.pt` 文件里保存了三项：`net` 模型参数、`acc` 准确率、`epoch` 轮数。


把这些步骤压缩成一句话就是：`main.py` 读参数，`data.py` 准备 CIFAR-10 batch，
`models/` 按名称创建网络，`train.py` 用交叉熵和 SGD 反复做前向传播、反向传播和参数更新，每个 epoch 后在测试集上评估，准确率创新高时把模型保存到 `weights/`。


## 其它文件/文件夹在实验里的作用
- `train_linear.py`：一个更简单的线性回归入门例子，用来理解“准备数据 -> 定义模型 -> 定义 loss -> 定义 optimizer -> forward/backward/update”的基本套路
- `dataset.py`：为推理/量化脚本准备数据，其中 `QDataset` 从 `data/q/*.jpg` 读取图片，文件名里的数字作为标签；
- `data/q/`：一组额外图片，主要给 `QDataset` 和后续量化校准/测试使用；
- `int8_infer.py`：加载 `weights/<model>.pt`，用 `torch2trt` 尝试转换 TensorRT INT8/FP16 推理，并在 `test.jpg` 上比较普通 PyTorch 模型和 TensorRT 模型的推理时间、预测类别；
- `test.jpg`：`int8_infer.py` 的单张测试图片；
- `/__pycache__/`：Python 自动生成的字节码缓存，可以不用关心。
 


# Questions
`data.py`
??? 通过补注释 我知道了这行代码大概做了什么，但我并不知道例如“RandomCrop对图像进行随机裁剪”到底有什么作用 (说看懂也感觉很虚), i.e., 如果不做会怎么样 / 对这些图像为什么需要这样的预处理流程/ 为什么这套预处理流程这样排序 / 换一套别的预处理流程会有什么样的效果
