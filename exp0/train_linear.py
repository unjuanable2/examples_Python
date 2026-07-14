import torch
# (语法) import 表示“导入模块”。
# torch 是 PyTorch 的主模块，里面有张量 tensor、模型训练、优化器等功能。

import torch.nn as nn
# torch.nn 是 PyTorch 里专门写神经网络结构的模块。
# as nn 表示给 torch.nn 起一个短名字 nn，后面写 nn.Linear、nn.MSELoss 会更方便。

import numpy as np
# numpy 是 Python 里常用的数值计算库。
# as np 表示给 numpy 起短名字 np。

import matplotlib.pyplot as plt
# matplotlib.pyplot 是画图模块。
# as plt 表示后面用 plt.plot、plt.show 画图。


################################################
# 1. 设置超参数                                 #
################################################
# 超参数：训练前由人手动设定的参数，不是模型自己学出来的。
# 例如学习率、训练轮数、输入维度、输出维度。

input_size = 1 # (变量) 保存输入特征的个数。
# 这里每个样本只有一个 x，所以 input_size = 1

output_size = 1 # (变量) 保存模型输出的个数。
# 这里每个样本只预测一个 y，所以 output_size = 1

num_epoches = 10 # (变量) 表示训练多少轮
# epoch：把整个训练集完整看一遍，叫 1 个 epoch。

learning_rate = 0.001 # (变量) 表示学习率: 控制每次根据梯度更新参数时“走多大一步”
# 太大可能震荡/不收敛，太小可能训练很慢。


################################################
# 2. 准备训练数据                               #
################################################
# 这个实验是线性回归，所以数据是一些成对的 (x, y)。
# 模型要学习的是：给一个 x，预测对应的 y。

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], 
        [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]], 
        dtype=np.float32)
# x_train 是变量，保存训练输入。
# - np.array(...) 表示创建一个 numpy 数组对象。
#   - 每个小列表例如 [3.3] 是一个训练样本的输入 x。
#   - dtype=np.float32 表示数组里的数字用 32 位浮点数保存。
#
# 为什么写成 [[3.3], [4.4], ...]，而不是 [3.3, 4.4, ...]？
# - [3.3, 4.4, ...] 的形状是 [15]，像一排数字；
# - [[3.3], [4.4], ...] 的形状是 [15, 1]，表示 15 个样本，每个样本 1 个特征；
# - nn.Linear(input_size, output_size) 期望输入通常是 [样本数, 特征数]。

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], 
        [2.596], [2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]], 
        dtype=np.float32)
# y_train 是变量，保存训练标签/真实答案。
# label：监督学习里，真实答案通常叫标签。
#
# 这里：
# 第 1 个 x 是 3.3，对应的真实 y 是 1.7；
# 第 2 个 x 是 4.4，对应的真实 y 是 2.76；
# 以此类推。


##############################################
# 3. 定义线性回归模型                           #
##############################################

model = nn.Linear(input_size, output_size)
# nn.Linear(input_size, output_size)：torch.nn 模块里的一个类，创建一个线性层对象/实例 y = w * x + b
#
# model 是变量，保存这个线性层对象。
# 这个对象内部有需要训练的参数：weight, bias (公式里的 b)
#
# 怎么知道有几个 w、几个 b？
# 看 nn.Linear(input_size, output_size) 这两个参数：
# - weight 的形状是 [output_size, input_size]
#   - 当前 weight 的形状是 [1, 1]，一共 1 * 1 = 1 个 w
#   - print(model.weight.shape)
# - bias 的形状是 [output_size]
#   - 当前 bias 的形状是 [1]，一共 1 个 b
#   - print(model.bias.shape)

################################################
# 4. 定义损失函数和优化器                       #
################################################
criterion = nn.MSELoss()
# criterion 是变量，保存损失函数对象。
# nn.MSELoss 是一个类，MSE 是 Mean Squared Error/ 平均平方误差
# - 对线性回归来说，MSELoss 是很常见的损失函数
#
# 它会计算：模型预测值 outputs 和 真实标签 targets 之间差多少。

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer 是变量，保存优化器对象。

# torch.optim.SGD(...) 创建的是随机梯度下降优化器。
# - model.parameters() 是 model 对象的方法调用。
#  把模型里需要训练的参数交给优化器，比如 weight 和 bias。
# - lr=learning_rate 是关键字参数
#   lr 是参数名，learning_rate 是前面定义的变量。
#   优化器之后会根据梯度和学习率更新这些参数。

loss_dict = [] # (变量) 保存一个列表 list。
# 后面每训练一轮，就把这一轮的 loss 数值放进这个列表，用来画 loss 曲线。

################################################
# 5. 迭代训练模型                               #
################################################
inputs = torch.from_numpy(x_train)
# torch.from_numpy(...) 把 numpy 数组转换成 PyTorch 张量 tensor。
# inputs 是变量，保存训练输入张量。
# PyTorch 模型不能直接训练 numpy 数组，一般要先转成 tensor。

# 这里可以放在 for 循环外面：
# 因为 x_train 这批训练输入在训练过程中不会改变，
# 所以不需要每个 epoch 都重新从 numpy 转一次 tensor。

targets = torch.from_numpy(y_train)
# targets 是变量，保存训练标签张量。
# inputs 和 targets 的样本顺序是一一对应的。

# 这里也可以放在 for 循环外面：
# 因为 y_train 这批真实答案也不会随着训练改变。

for epoch in range(num_epoches):
    # for ... in ... 是循环语句。
    # range(num_epoches) 会生成 0, 1, 2, ..., num_epoches-1。
    # epoch 是循环变量，表示当前是第几轮训练。

    outputs = model(inputs)
    # model(inputs) 是函数调用形式，但这里更准确地说是“调用模型对象做前向传播”。
    # 这一行必须放在 for 循环里面，不能提前到循环外面只做一次。
    #
    # 原因：
    # - inputs 和 targets 是固定不变的数据；
    # - 但 model 里的 weight、bias 会在每个 epoch 的 optimizer.step() 后更新；
    # - 参数更新以后，同样的 inputs 会得到新的 outputs；
    # - 所以每一轮都要重新做一次前向传播，得到“当前模型”的预测结果。
    
    # outputs 是变量，保存模型预测结果。
    # 对每一个输入 x，模型都会根据当前的 w 和 b 预测一个 y。

    loss = criterion(outputs, targets)
    # criterion(outputs, targets) 调用损失函数对象。
    # 它会比较 outputs (模型预测值) 和 targets (真实标签), 然后算出一个 loss。
    
    # loss 是变量，保存这一次前向传播得到的损失值。
    # - 越小，说明模型预测越接近真实答案。

    optimizer.zero_grad()
    # zero_grad() 是 optimizer 对象的方法。
    # 作用：清空上一轮训练留下的梯度。
    # PyTorch 默认会累加梯度，所以每一轮反向传播前通常都要先清空。

    loss.backward()
    # backward() 是 loss 对象的方法。
    # 作用：反向传播。
    # 根据当前 loss，计算模型参数 weight、bias 各自应该往哪个方向改。
    # 这些“应该怎么改”的信息叫梯度 gradient。

    optimizer.step()
    # step() 是 optimizer 对象的方法。
    # 作用：根据梯度修改 weight 和 bias

    loss_dict.append(loss.item())
    # append(...) 是列表对象的方法，表示往列表末尾添加一个元素。
    # loss.item() 会把只有一个数的 PyTorch tensor 转成普通 Python 数字。
    # 这里把每个 epoch 的 loss 保存下来，后面用来画 loss 曲线。

    if (epoch + 1) % 5 == 0:
        # epoch 从 0 开始，所以 epoch + 1 才是人习惯看的“第几轮”。
        # (epoch + 1) % 5 == 0 表示每 5 轮打印一次。

        print('Epoch [{}]/[{}], loss: {:.4f}'.format(epoch + 1, num_epoches, loss.item()))
        # print(...) 是打印到终端。
        # format(...) 是字符串对象的方法，用来把变量填进字符串。
        # {:.4f} 表示把 loss 保留 4 位小数显示。


################################################
# 6. 画出模型拟合结果                           #
################################################
predicted = model(torch.from_numpy(x_train)).detach().numpy()
# 这一行用训练后的 model 再预测一次所有 x_train。
# predicted 是变量，保存模型最终预测出的 y。
# - model(torch.from_numpy(x_train))：把 x_train 转成 tensor 后送进模型；
# - detach()：把预测结果从“继续参与训练/反向传播”的状态里拿出来，
#             i.e., 这个 predicted 只是拿来看的普通结果，不要再跟踪它的梯度关系
#   - 如果不 detach() 就直接 .numpy()，PyTorch 通常会报错，
#     因为它不希望你把一个还连着梯度计算图的 tensor 直接转成 numpy。
# - numpy()：把 tensor 转回 numpy 数组，方便 matplotlib 画图。

plt.plot(x_train, y_train, 'ro', label='Original data')
# plt.plot(...) 是画图函数。
# - x_train 是横坐标，y_train 是纵坐标。
# - 'ro' 表示 red circle，也就是红色圆点。
# - label='Original data' 是图例名称。

plt.plot(x_train, predicted, label='Fitted line')
# 画出模型拟合出来的线。

plt.legend()
# 显示图例，也就是 Original data / Fitted line 这些文字说明。

plt.savefig('train_linear_fit.png')
# savefig(...) 是 plt 提供的保存图片函数。
# 这里会把第一张图保存成当前目录下的 train_linear_fit.png。
# （如果你在 exp0 目录里运行，图片就会保存到 exp0/train_linear_fit.png。）

# savefig 要放在 show() 前面更稳妥：
# 因为有些环境里 show() 结束后，当前图像可能会被清空或关闭。

plt.show()
# 显示第一张图：原始数据点和模型拟合线。


################################################
# 7. 画出 loss 变化曲线                         #
################################################
plt.plot(loss_dict, label='loss for every epoch')
# 横轴默认是列表下标，也就是第几个 epoch；
# 纵轴是每个 epoch 保存下来的 loss。
# 如果训练正常，loss 通常会逐渐下降。

plt.legend()
# 显示图例。

plt.savefig('train_linear_loss.png')
# 保存第二张图：loss 随 epoch 变化的曲线。
# 如果你在 exp0 目录里运行，图片会保存到 exp0/train_linear_loss.png。

plt.show()
# 显示第二张图：loss 随训练轮数变化的曲线。
