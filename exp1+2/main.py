from __future__ import print_function
# (语法) 上面这一行是兼容写法：让 Python2 也使用 Python3 风格的 print()。
#       如果只学 Python3，可以先理解为“历史兼容代码”，不是训练流程的核心。

import argparse
# (语法) import 表示“导入模块”。
# (语法) 模块可以理解成一个 .py 文件或一组代码包，里面放了别人写好的类、函数、变量等。

# argparse 是 Python 标准库里的一个模块，专门用来解析命令行参数。

import torch
# torch 是 PyTorch 的主模块，里面包含张量、GPU、神经网络训练等功能。

from data import testloader, trainloader
# 从 data.py 中导入两个变量：
# - trainloader：训练集的数据加载器对象
# - testloader：测试集的数据加载器对象

# 这里的 data 不是 data/ 文件夹。
# 判断依据：data.py 里定义了 testloader 和 trainloader；
# data/ 文件夹只是存放图片数据，当前没有作为 Python 包被导入。

from models import model_factory
# 从 models 包中导入 model_factory: 一个函数, 给它一个模型名称字符串，它返回对应的模型对象。

# (语法) 包: 一个文件夹，
#   里面有一个 __init__.py 文件：表示这个文件夹是一个 Python 包。
#   里面可以有很多 .py 文件，里面放了类、函数、变量等。

from train import Trainer
# 从 train.py 中导入 Trainer: 一个类。

# (语法) 类可以理解成“对象的模板”，规定对象有哪些数据和行为。
#   - 类：设计图，例如 ArgumentParser；
#   - 对象/实例：按设计图造出来的具体东西，例如 parser。
#     - 对象/实例在这里基本可以当成同一个意思理解

################################################
# 解析命令行参数                                 #
################################################
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # ArgumentParser 是 argparse 模块里的一个类；
    # argparse.ArgumentParser(...) 表示“调用这个类的构造方法，创建一个对象”；

    # description='...' 是关键字参数，用来给这个命令行程序写说明文字。
    # description 是参数名，'PyTorch CIFAR10 Training' 是传进去的参数值。
    # (语法) 关键字参数：是函数/方法调用时，指定参数名和参数值的写法。
    #   - 参数可以影响这次调用的结果。这里 description 会影响 --help 里显示的说明文字。
    #   - 对比：不写参数名、只按顺序传值的写法叫“位置参数”，例如 print('hello')。

    # parser 是变量名，保存刚刚创建出来的 ArgumentParser 对象

parser.add_argument('--lr', default=0.1, type=float, help='learning Rate')
# parser 是 ArgumentParser 对象，add_argument 是它提供的方法（属于某个对象的函数）
# parser.add_argument(...) 是“调用对象的方法”
#
# 这一行告诉 parser：程序支持一个叫 --lr 的命令行参数。
# - '--lr'：参数名字，运行时可以写 python main.py --lr 0.2；
# - default=0.1：如果用户不写 --lr，就默认使用 0.1；
# - type=float：把命令行读到的文本转换成浮点数 float；
# - help='...'：命令行帮助信息。

parser.add_argument('--steps', '-n', default=200, type=int, help='No of Steps')
# 这一行添加训练轮数参数。
# - '--steps' 和 '-n' 是同一个参数的两种写法
# - type=int 表示把输入转换成整数 int。

parser.add_argument('--gpu', '-p', action='store_true', help='Train on GPU')
# 这一行添加是否使用 GPU 的开关参数。
# - action='store_true' 表示：
#   - 如果命令行里写了 --gpu 或 -p，args.gpu 就是 True；
#   - 如果没写，args.gpu 就是 False。

parser.add_argument('--fp16', action='store_true', help='Train with FP16 weights')
# 这一行添加 FP16 半精度训练开关。
parser.add_argument('--loss_scaling', '-s', action='store_true', help='Scale FP16 losses')
# 这一行添加 loss scaling 开关，主要配合 FP16 使用。

parser.add_argument('--model', '-m', default='resnet18', type=str, help='Name of Network')
# 这一行添加模型名称参数。
# - default='resnet18' 表示默认模型名是字符串 'resnet18'。
# - type=str 表示把输入当作字符串 str。


args = parser.parse_args()
# parse_args() 也是 parser 对象的方法：会真正读取终端命令里的参数，并把结果放进 args 这个变量。
# args 是一个对象，里面会有 lr、steps、gpu、fp16、loss_scaling、model 等属性。
#
# 例如运行：
#   python main.py --lr 0.2 --steps 200 --gpu --model resnet18
# 大致会得到：
#   args.lr == 0.2
#   args.steps == 200
#   args.gpu == True
#   args.model == 'resnet18'
#
# (语法) args.lr 这种写法叫“访问对象的属性”。

train_on_gpu = False
if args.gpu and torch.cuda.is_available():
    # torch.cuda.is_available() 是函数调用，用来检查当前机器是否真的有可用 CUDA GPU。
    # (语法) and 表示“并且”：左右两边都为 True，整个条件才为 True。

    train_on_gpu = True
    
    # 如果想用 FP16 半精度训练，需要启用 CuDNN。
    # 原因：
    # - FP16（比普通 FP32 更省显存，计算也可能更快）训练依赖 GPU/CuDNN 对半精度卷积和相关算子的支持，
    #   否则代码后面的半精度路径会被关闭。 
    # - 神经网络里的卷积、BatchNorm 等操作不是 Python 自己算的，它们通常会交给 GPU 后端库去高效执行；
    # - CuDNN 是 NVIDIA 给深度学习准备的 GPU 加速库，对卷积等操作有专门优化，也支持半精度相关计算；
    #
    # 对应 train.py 里的逻辑是：
    # if train_on_gpu and torch.backends.cudnn.enabled:
    #     self.fp16_mode = fp16
    # else:
    #     self.fp16_mode = False
    # 也就是说，只有“使用 GPU”并且“CuDNN 已启用”时，FP16 模式才可能真正打开。
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn 是 PyTorch 里和 CuDNN 后端相关的配置对象，
    # enabled 和 benchmark 是这个配置对象的属性。
    # 启用 CuDNN benchmark：让 PyTorch/CuDNN 在运行 ResNet18 的卷积层时，决定在 GPU 上怎么更快地算
    
    # 这里容易混淆两个“选择”：
    # 1. model_name = 'resnet18' 选择的是“模型结构”(网络有哪些层、层怎么连接) 
    #    e.g., ResNet18 里有卷积层、BN 层、残差连接、全连接层。
    # 2. torch.backends.cudnn.benchmark = True 选择的是“底层计算实现”。
    #    每个卷积层在数学上都是同一种操作：输入特征图和卷积核做卷积，输出新的特征图。
    #    但在 GPU 上，完成同一个卷积操作可以有多种底层算法/实现方式,
    #    例如不同算法可能在速度、显存占用、适合的输入尺寸上不同。


######################################################
# 训练流程的核心部分：创建模型、创建训练器、开始训练          #
######################################################
model_name = args.model
# model_name 是变量，保存用户指定的模型名称。
# 如果用户没有指定，args.model 就是默认值 'resnet18'。

model = model_factory(model_name)
# model_factory 来自 models 包
# model_factory(model_name) 是函数调用: 根据字符串 model_name 创建一个神经网络模型对象。
# e.g. model_name == 'resnet18' 时，它会创建 ResNet18 模型。

# model 是变量，保存这个模型对象。

trainer = Trainer(model_name, model,
    args.lr, train_on_gpu, args.fp16, args.loss_scaling)
# Trainer 是 train.py 里定义的一个类
# Trainer(...) 表示调用 Trainer 这个类，创建一个 Trainer 对象/实例。
# 括号里的内容是传给 Trainer 构造方法的参数：
# - model_name：模型名字
# - model：模型对象
# - args.lr：学习率
# - train_on_gpu：是否使用 GPU
# - args.fp16：是否使用 FP16
# - args.loss_scaling：是否使用 loss scaling

# trainer 是变量，保存这个训练器对象。

trainer.train_and_evaluate(trainloader, testloader, args.steps)
# trainloader 和 testloader 是 data.py 里创建的两个数据加载器对象。

# trainer.train_and_evaluate(...) 是调用 trainer 对象的方法 (在 train.py 中定义)
# 这个方法会开始完整训练流程：
# - 使用 trainloader 读取训练数据并更新模型参数；
# - 使用 testloader 读取测试数据并评估准确率；
# - args.steps 决定训练多少个 epoch。
