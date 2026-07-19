# exp2：AlexNet CIFAR-10 训练实验

本实验在与 exp1 相同的 CIFAR-10 数据和训练框架上，使用 `models/alexnet.py` 中的 `AlexNet(nn.Module)` 训练 200 个 epoch，并记录训练和测试准确率。


## 文件
- 数据集统一从 `../exp1/data/` 读取；如果数据尚不存在，torchvision 也会下载到该目录，因此不需要 `./data/`。
- `models/alexnet.py`：AlexNet 模型的 `__init__()` 和 `forward()` 实现。
- `run_exp.sh`：检查 CUDA GPU 并启动 200 epoch 训练。
  - 在启动训练前会输出 CUDA GPU 名称；如果没有检测到 CUDA GPU，脚本会直接退出，不会改用 CPU 训练。
- `./results_analysis`
  - `analyze_exp2_log.py`：日志分析脚本，负责从训练日志生成 CSV 和曲线图
  - `run_exp2_out.txt`：完整训练日志，包含每个 epoch 的 loss 和训练/测试准确率。
  - `exp2_epoch_metrics.csv`：从日志自动提取的逐 epoch 指标表，字段与实验 1 的 csv 一致。
  - `exp2_accuracy_curve.png` / `.svg`：训练和测试准确率曲线。
  - `exp2_loss_curve.png` / `.svg`：训练和测试 loss 曲线。
- `./weights/alexnet/`：测试准确率刷新时保存的 AlexNet 权重。

## 结果 和 分析
- 根据 `exp2_epoch_metrics.csv` 的 200 个 epoch，AlexNet 
  - 训练准确率从 `10.060%` 提升到 `86.284%`。最佳训练准确率为 `86.540%`（epoch `187`）
  - 测试准确率在 epoch `37` 首次达到 `82%`，之后基本稳定在 `83%` 左右，最后一个 epoch 为 `82.990%`。最佳测试准确率为 `83.050%`（epoch `131`）
- 相较于 ResNet（exp1）最佳测试准确率 91.8%， AlexNet 最佳测试准确率 83.05%，差距约 8.7 个百分点
  - 主要原因是 ResNet 的残差连接让网络更容易训练，并能提取更深、更有效的特征。AlexNet 是较早期的架构，特征提取能力通常弱于 ResNet。
