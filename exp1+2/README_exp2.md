# exp2：AlexNet CIFAR-10 训练实验

本实验在与 exp1 相同的 CIFAR-10 数据和训练框架上，使用 `models/alexnet.py` 中的 `AlexNet(nn.Module)` 训练 200 个 epoch，并记录训练和测试准确率。

## 运行

本实验必须在配有 NVIDIA CUDA GPU 的环境中运行：

```bash
chmod +x run_exp2.sh
./run_exp2.sh
```

如果 Python 虚拟环境不在默认的 `~/pyenv`：

```bash
VENV_PATH=/path/to/venv ./run_exp2.sh
```

脚本实际执行的训练参数为：

```bash
python main.py --model alexnet --steps 200 --lr 0.1 --gpu
```

## 文件
- `models/alexnet.py`：AlexNet 模型的 `__init__()` 和 `forward()` 实现。
- `run_exp2.sh`：检查 CUDA GPU 并启动 200 epoch 训练。
  - 在启动训练前会输出 CUDA GPU 名称；如果没有检测到 CUDA GPU，脚本会直接退出，不会改用 CPU 训练。
- `results_analysis_exp2/run_exp2_out.txt`：完整训练日志，包含每个 epoch 的 loss 和训练/测试准确率。
- `results_analysis_exp2/exp2_epoch_metrics.csv`：从日志自动提取的逐 epoch 指标表，字段与实验 1 的 `exp1_epoch_metrics.csv` 一致。
- `results_analysis_exp2/exp2_accuracy_curve.png` / `.svg`：训练和测试准确率曲线。
- `results_analysis_exp2/exp2_loss_curve.png` / `.svg`：训练和测试 loss 曲线。
- `weights/alexnet/`：测试准确率刷新时保存的 AlexNet 权重。


## 结果
