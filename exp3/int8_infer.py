from torchvision import transforms, datasets
# torchvision.transforms 提供图像预处理操作，例如缩放、转 Tensor 和标准化。
# datasets 在当前脚本中没有直接使用，保留它是为了兼容原实验代码。

import torch
# torch 是 PyTorch 主模块，负责张量计算、加载模型权重以及调用 CUDA GPU。

import torch.nn as nn
# torch.nn 提供神经网络组件；当前文件没有直接使用 nn，但模型内部会使用它。

from dataset import QDataset, testloader
# 从 exp3/dataset.py 导入：
# - QDataset：INT8 校准数据集类；
# - testloader：CIFAR-10 测试集的数据加载器。

from PIL import Image, ImageDraw, ImageFont, ImageFilter
# PIL 用于读取测试图片并在图片上绘制分类结果：
# - Image：打开、保存图片；
# - ImageDraw：在图片上写字；
# - ImageFont：加载字体；
# - ImageFilter：当前脚本没有直接使用。

from torch2trt import torch2trt  # tensorRT
# torch2trt 会读取 PyTorch 模型和示例输入，并将模型转换为 TensorRT 引擎。
# 转换后的对象仍可像普通 PyTorch 模型一样通过 model_trt_int8(x) 调用。

import argparse
# argparse 用来解析命令行参数，例如 -p 和 -m alexnet。

from models import model_factory
# model_factory 根据模型名称字符串创建对应的 PyTorch 网络对象。

import time
# time 用于记录一次推理的运行时间。

import tensorrt as trt
# trt 提供 TensorRT 的类型和配置项。当前生效代码没有直接访问 trt；
# 下方被注释掉的校准算法配置会用到它。

import os
# os.path.join 用于拼接测试图片路径。


def test(testmodel):
    """在完整 CIFAR-10 测试集上计算给定模型的分类准确率。"""

    correct = 0
    # correct：到目前为止预测正确的图片数量。

    total = 0
    # total：到目前为止已经测试的图片总数。

    with torch.no_grad():
        # 测试阶段不需要反向传播。
        # no_grad() 会关闭梯度记录，减少显存占用和额外计算。

        for batch_idx, (inputs, targets) in enumerate(testloader):
            # testloader 每次返回一个 batch：
            # - inputs：图片张量，形状大致是 [batch_size, 3, 32, 32]；
            # - targets：每张图片对应的类别编号，形状大致是 [batch_size]；
            # - batch_idx：当前是测试集中的第几个 batch。

            inputs, targets = inputs.cuda(), targets.cuda()
            # 将图片和标签从 CPU 内存复制到 CUDA GPU 显存。

            outputs = testmodel(inputs)
            # 前向传播。outputs 的形状是 [batch_size, 10]，
            # 每一行包含该图片属于 CIFAR-10 十个类别的原始分数（logits）。

            _, predicted = outputs.max(1)
            # max(1) 沿类别维度寻找最大值：
            # - 第一个返回值是最大分数，这里用 _ 表示不需要保存；
            # - predicted 是最大分数所在的类别编号。

            total += targets.size(0)
            # targets.size(0) 是当前 batch 的实际图片数量。

            correct += predicted.eq(targets).sum().item()
            # predicted.eq(targets) 逐项比较预测类别和真实类别；
            # sum() 统计当前 batch 预测正确的数量；item() 转为 Python 数值。

    acc = 100. * correct / total
    # 用百分数表示准确率，例如 83.05 表示 83.05%。

    return acc


if __name__ == "__main__":
    # 只有直接执行 python int8_infer.py 时才运行下面的主程序。
    # 如果其他文件 import int8_infer，下面的推理流程不会自动执行。

    ################################################
    # 解析命令行参数                               #
    ################################################
    parser = argparse.ArgumentParser(description='Test for Cifar10 w/ or w/o trt')
    # 创建命令行参数解析器。description 会显示在 --help 帮助信息中。

    parser.add_argument('--gpu', '-p', action='store_true', help='Trained on GPU')
    # --gpu 和 -p 是同一个布尔开关。
    # action='store_true' 表示命令中写了 -p 时 args.gpu 为 True，否则为 False。

    parser.add_argument('--model', '-m', default='alexnet', type=str, help='Name of Network')
    # 指定网络名称；未提供 -m 时默认使用 alexnet。

    args = parser.parse_args()
    # 读取终端命令并把解析结果保存在 args 对象中。

    ################################################
    # 创建模型并加载训练好的权重                   #
    ################################################
    model_name = args.model
    # model_name 保存用户指定的模型名，例如 'alexnet'。

    model = model_factory(model_name)
    # 根据模型名称创建网络结构。此时只有网络结构，还没有加载训练结果。

    print("Testing model: %s" % model_name)

    if args.gpu and torch.cuda.is_available():
        # 当前 TensorRT 转换和推理依赖 NVIDIA CUDA GPU，所以必须同时满足：
        # 1. 用户在命令行中写了 -p/--gpu；
        # 2. PyTorch 检测到可用的 CUDA GPU。

        torch.backends.cudnn.enabled = True
        # 启用 NVIDIA CuDNN 深度学习加速库。

        torch.backends.cudnn.benchmark = True
        # 让 CuDNN 针对固定的输入尺寸寻找较快的卷积实现。
        # 本实验输入始终是 1×3×32×32，适合开启 benchmark。

        model = model.cuda()
        # 将模型参数和缓冲区复制到 GPU。
    else:
        print("-p is a must when executing this script. Exiting...")
        # 没有 -p 或没有 CUDA 时无法继续进行 TensorRT 推理。
        exit();

    model.load_state_dict(torch.load('./weights/' + model_name + '.pt')['net'])
    # torch.load(...) 读取训练阶段保存的 checkpoint 字典；
    # ['net'] 取出模型参数，load_state_dict(...) 将参数载入刚创建的网络。

    accbefore = torch.load('./weights/' + model_name + '.pt')['acc']
    # ['acc'] 是 checkpoint 中保存的准确率。
    # 当前单图流程不会打印它；下方被三引号注释的整集测试原本会使用该变量。

    ################################################
    # 定义模型输入所需的图像预处理                 #
    ################################################
    cali_augmentation = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # 随机裁剪适合训练时做数据增强，但校准和推理需要稳定、可重复的输入，
        # 所以这里将它注释掉。

        transforms.Resize((32, 32), interpolation=Image.BICUBIC),
        # 将输入图像缩放到 CIFAR-10 模型要求的 32×32 像素。

        transforms.ToTensor(),
        # 把 PIL 图片转换为 PyTorch 张量：形状变为 [3, 32, 32]，
        # 同时把像素值从 0～255 缩放到 0～1。

        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # 按 RGB 三个通道分别执行 (像素值 - mean) / std。
        # 这些均值和标准差与 CIFAR-10 训练时使用的预处理保持一致。
    ])

    cali_cifar10 = QDataset(transform=cali_augmentation)
    # 创建用于 INT8 校准的数据集对象。
    # 当前 torch2trt 调用中的 int8_calib_dataset 参数被注释，因此它暂未生效。

    model.eval()
    # 切换到评估模式，使 BatchNorm 使用已学习的统计量，并关闭 Dropout 的随机丢弃。

    ################################################
    # 将 PyTorch 模型转换成 TensorRT INT8 引擎     #
    ################################################
    x = torch.randn([1, 3, 32, 32]).cuda()  # no .half()?
    # 创建示例输入，形状为 [1, 3, 32, 32]：
    # - 1：batch size，一次输入一张图片；
    # - 3：RGB 三个颜色通道；
    # - 32×32：图片尺寸。
    # torch2trt 会通过这个输入追踪计算图并确定各层张量形状。
    # 这里不用 .half()；转换参数会决定 TensorRT 引擎采用的精度。

    model_trt_int8 = torch2trt(model, [x],
                          fp16_mode=True,
                          int8_mode=True)
    # 把 PyTorch 模型转换成 TensorRT 模型：
    # - [x]：模型示例输入必须放在列表中；
    # - fp16_mode=True：允许 TensorRT 对支持的层使用 FP16；
    # - int8_mode=True：启用 INT8 量化执行。
    #
    # INT8 用 8 位整数近似表示部分激活值和权重，通常能降低内存/带宽开销并加速推理，
    # 但量化误差可能使准确率略微下降。实际采用哪种精度仍由 TensorRT 按层选择。

                    #      max_batch_size=1,
                    # max_batch_size=1 表示引擎最多一次处理一张图片。
                     #     int8_calib_dataset=cali_cifar10,
                     # 指定代表性图片做 INT8 校准，以确定浮点值到整数值的缩放范围。
                      #    int8_calib_algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2)
                      # 指定 TensorRT 的熵校准算法。以上参数目前都被注释，不会执行。

    '''
    这一段被三引号包围，因此当前程序不会执行。
    如果取消注释，它会在完整测试集上分别测量原模型和 TensorRT 模型的准确率与总耗时。

    start_time = time.time()
    accbefore = test(model)
    torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
    time_spent = time.time() - start_time
    print("Acc for fp32: %.2f; Time spent: %.8fms" % (accbefore, time_spent))

    start_time = time.time()
    accafter = test(model_trt_int8)
    torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
    time_spent = time.time() - start_time
    print("Acc for trt int8: %.2f; Time spent: %.8fms" % (accafter, time_spent))

    '''

    ################################################
    # 读取一张图片并进行 FP32/INT8 推理对比        #
    ################################################
    test_image = os.path.join('test.jpg')
    # 当前写法等价于 test_image = 'test.jpg'。
    # 该相对路径以运行脚本时的当前工作目录为基准。

    img = Image.open(test_image)
    # 打开测试图片并得到 PIL Image 对象。

    img_tensor = cali_augmentation(img)
    # 应用与训练一致的缩放、转张量和标准化。
    # 此时 img_tensor 的形状是 [3, 32, 32]，还没有 batch 维度。

    # print(img_tensor)
    img_tensor = img_tensor.unsqueeze_(0).cuda()
    # unsqueeze_(0) 在最前面原地增加 batch 维度：
    # [3, 32, 32] -> [1, 3, 32, 32]，然后将张量复制到 GPU。

    # print(img_tensor)
    # input = img_tensor.cuda()

    start_time = time.time()
    # 记录 FP32 推理开始前的 CPU 时间。

    y_fp32 = model(img_tensor)
    # 使用原始 PyTorch 模型前向推理，输出形状为 [1, 10]。

    torch.cuda.synchronize()
    # CUDA 运算默认异步执行。synchronize() 等待 GPU 完成本次推理，
    # 防止 CPU 提前读取结束时间而低估实际耗时。

    time_spent_fp32 = time.time() - start_time
    # 计算 FP32 单次推理耗时，单位是秒。

    print('Time Spent for fp32: {:.2f}ms'.format(time_spent_fp32 * 1000))
    # 乘以 1000 将秒转换为毫秒。

    start_time = time.time()
    y_int8 = model_trt_int8(img_tensor)
    # 使用转换后的 TensorRT INT8 引擎对同一张图片推理。

    torch.cuda.synchronize()
    time_spent_int8 = time.time() - start_time
    print('Time Spent for int8: {:.2f}ms'.format(time_spent_int8 * 1000))

    # print(y)
    percentage_fp32 = torch.softmax(y_fp32[0], dim=0) * 100
    percentage_int8 = torch.softmax(y_int8[0], dim=0) * 100  # 得到概率
    # y_fp32[0]/y_int8[0] 取 batch 中第一张图片的十个 logits。
    # softmax 将十个原始分数转换为总和为 1 的相对概率，再乘 100 得到百分比。

    # print(percentage)
    cl_fp32, index_fp32 = torch.max(percentage_fp32, 0)  # 概率最大值，模型认为的物体
    cl_int8, index_int8 = torch.max(percentage_int8, 0)
    # torch.max(..., 0) 返回：
    # - cl_*：最高类别概率；
    # - index_*：最高概率对应的类别编号，即模型预测结果。

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # CIFAR-10 的类别编号与英文类别名映射。

    font = ImageFont.truetype('LiberationSans-Regular.ttf', 30)
    # 从当前工作目录加载字体文件，字号为 30。

    draw = ImageDraw.Draw(img)
    # 创建绘图对象，后续文字会直接画到原始 img 上。

    text = 'Mode: fp32, ' + '{:.2}ms '.format(time_spent_fp32 * 1000) + str(classes[index_fp32]) + ' (' + '{:.2f}'.format(cl_fp32.item()) + '%' + ')'
    # 拼接 FP32 结果文字，包括推理耗时、预测类别和最高概率。
    # PyTorch 的零维整数张量可以作为列表索引使用；item() 把概率张量转成 Python 数值。

    draw.text((0, 0), text, font=font, fill="#f000ff", spacing=0, align='left')
    # 在图片左上角坐标 (0, 0) 写入 FP32 结果。

    print(text + '\n')

    text = 'Mode: int8, ' + '{:.2}ms '.format(time_spent_int8 * 1000) + str(classes[index_int8]) + ' (' + '{:.2f}'.format(cl_int8.item()) + '%' + ')'
    draw.text((0, 40), text, font=font, fill="#ff00ff", spacing=0, align='left')
    # 在 y=40 的位置写入 TensorRT INT8 结果，避免与第一行重叠。

    img.save(test_image, 'jpeg')
    # 将带有两行结果文字的图片保存回 test.jpg，会覆盖原文件内容。

    print(text)
    # 在终端输出 INT8 的推理结果。
