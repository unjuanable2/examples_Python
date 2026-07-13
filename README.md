# Intern1

- `./exp0/` 用最简单的线性模型练习 PyTorch 训练流程：准备输入数据和标签，定义模型，计算 loss，反向传播 backward，调用优化器 optimizer 更新参数。
  - 这个实验主要用来理解深度学习训练代码的基本结构。 
- `./exp1/` CIFAR-10 图像分类训练实验: 用 PyTorch 在 CIFAR-10 数据集上训练 ResNet18 等 CNN 分类模型。代码包含数据读取和增强、模型选择、训练循环、测试集评估、学习率调整、FP16 混合精度训练、模型权重保存等完整训练流程。
- `./exp2/` TensorRT / INT8 推理部署实验: 基于训练好的模型，尝试把 PyTorch 模型转换成 TensorRT 推理模型，并比较普通 PyTorch 推理和 TensorRT 推理的速度。实验里还包含 INT8 量化校准数据集，用来理解模型部署和推理加速。
