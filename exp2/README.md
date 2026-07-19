# exp2：AlexNet CIFAR-10 训练实验

本实验在与 exp1 相同的 CIFAR-10 数据和训练框架上，使用 `models/alexnet.py` 中的 `AlexNet(nn.Module)` 训练 200 个 epoch，并记录训练和测试准确率。


## 文件
- 数据集统一从 `../exp1/data/` 读取；如果数据尚不存在，torchvision 也会下载到该目录，因此不需要 `./data/`。
- `models/alexnet.py`：AlexNet 模型的 `__init__()` 和 `forward()` 实现。
- `run_exp.sh`：检查 CUDA GPU 并启动 200 epoch 训练。
  - 在启动训练前会输出 CUDA GPU 名称；如果没有检测到 CUDA GPU，脚本会直接退出，不会改用 CPU 训练。
- `./results_analysis`
  - `run_exp2_out.txt`：完整训练日志，包含每个 epoch 的 loss 和训练/测试准确率。
  - `exp2_epoch_metrics.csv`：从日志自动提取的逐 epoch 指标表，字段与实验 1 的 csv 一致。
  - `exp2_accuracy_curve.png` / `.svg`：训练和测试准确率曲线。
  - `exp2_loss_curve.png` / `.svg`：训练和测试 loss 曲线。
- `./weights/alexnet/`：测试准确率刷新时保存的 AlexNet 权重。

## 结果
