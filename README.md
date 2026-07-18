# Intern1

- `./exp0/` 用最简单的线性模型练习 PyTorch 训练流程：准备输入数据和标签，定义模型，计算 loss，反向传播 backward，调用优化器 optimizer 更新参数。
  - 这个实验主要用来理解深度学习训练代码的基本结构。 
- `./exp1+2/` CIFAR-10 图像分类训练实验：共用一套 PyTorch 训练框架。
  - exp1 使用 ResNet18 模型训练 200 个 epoch，说明和脚本分别为 `README_exp1.md` 和 `run_exp1.sh`, 结果保存在 `results_analysis_exp1/`
  - exp2 实现并训练 AlexNet 200 个 epoch，说明和脚本分别为 `README_exp2.md` 和 `run_exp2.sh`, 结果保存在 `results_analysis_exp2/`
- `./exp3/` TensorRT / INT8 推理部署实验：基于实验 2 训练好的 AlexNet，尝试把 PyTorch 模型转换成 TensorRT 推理模型，并比较 PyTorch FP32 与 TensorRT INT8 的推理速度和预测结果。
