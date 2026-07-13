
## 其它文件/文件夹在实验里的作用
- `train_linear.py`：一个更简单的线性回归入门例子，用来理解“准备数据 -> 定义模型 -> 定义 loss -> 定义 optimizer -> forward/backward/update”的基本套路
- `dataset.py`：为推理/量化脚本准备数据，其中 `QDataset` 从 `data/q/*.jpg` 读取图片，文件名里的数字作为标签；
- `data/q/`：一组额外图片，主要给 `QDataset` 和后续量化校准/测试使用；
- `int8_infer.py`：加载 `weights/<model>.pt`，用 `torch2trt` 尝试转换 TensorRT INT8/FP16 推理，并在 `test.jpg` 上比较普通 PyTorch 模型和 TensorRT 模型的推理时间、预测类别；
- `test.jpg`：`int8_infer.py` 的单张测试图片；
- `/__pycache__/`：Python 自动生成的字节码缓存，可以不用关心。