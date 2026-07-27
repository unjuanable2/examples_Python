from torchvision import transforms
# torchvision.transforms 是 PyTorch 的图像预处理模块，提供 Resize、ToTensor、
# Normalize 等操作。后面会用 transforms.Compose(...) 把这些操作按顺序组合起来。
#
# (语法) from torchvision import transforms 表示从 torchvision 包中导入 transforms，
# 因此后面可以直接写 transforms.Resize，而不用写 torchvision.transforms.Resize。

import torch
# torch 是 PyTorch 主模块，本文件用它完成：
# - 创建和计算张量；
# - 加载模型 checkpoint；
# - 把模型与图片移动到 CUDA GPU；
# - 执行 softmax、取最大概率以及等待 CUDA 完成。

from data import QDataset, testloader
# 这里的 data 指当前 exp3 目录中的 data.py，不是 data/ 图片文件夹。
# 从 data.py 导入两个已经定义好的对象：
# - QDataset：INT8 校准数据集类；
# - testloader：CIFAR-10 测试集的数据加载器。

from PIL import Image, ImageDraw, ImageFont, ImageFilter
# PIL 是 Python 图像处理库。本行一次导入四个名字：
# - Image：打开、保存图片；
# - ImageDraw：在图片上写字；
# - ImageFont：加载字体；
# - ImageFilter：当前文件没有使用，可以删除，不影响程序功能。

from torch2trt import torch2trt  # tensorRT
# torch2trt 是一个函数：接收 PyTorch 模型、示例输入和精度配置，
# 返回转换后的 TensorRT 模型对象。
# 转换后仍然可以像调用普通模型一样写 model_trt_int8(input_tensor)。

import argparse
# argparse 是 Python 标准库，用来解析命令行参数


from pathlib import Path # Path 用对象表示文件路径
EXP3_DIR = Path(__file__).resolve().parent
# __file__ 是当前 int8_infer.py 的路径
#
# resolve().parent 得到 exp3 目录。
EXP2_DIR = EXP3_DIR.parent / "exp2"
# exp3 的上一级是项目根目录，再拼接 exp2，得到 exp2 的绝对路径。
#
import sys # sys 是 Python 标准库。
sys.path.insert(0, str(EXP2_DIR))
# sys.path 保存 Python 搜索模块的目录列表。
#
# insert(0, ...) 把 exp2 放在搜索顺序的最前面；
# 执行 from models import ... 时，Python 就会进入 exp2/models，而不会误用 exp1/models 或其它同名包。
#
# str(...) 把 Path 对象转换成 sys.path 接收的路径字符串。
#
from models import model_factory
# 由于上面已经把 EXP2_DIR 放到 sys.path 最前面，这里的 models 明确指 exp2/models。
# model_factory 来自 exp2/models/model_factory_dict.py：
# 传入模型名称字符串，例如 "alexnet"，返回对应的 PyTorch 模型对象。


import time
# time 是 Python 标准库，
# 本文件使用 time.time() 读取 CPU 墙钟时间，从而计算 FP32 和 TensorRT 模型各进行一次推理所花的时间。

import tensorrt
# tensorrt 提供 TensorRT 的配置类型。
# 下面通过 tensorrt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
# 指定 INT8 activation values 使用熵校准算法确定量化范围。


################################################
# 在完整 CIFAR-10 测试集上计算给定模型的分类准确率   #
################################################
def test(testmodel):
    # testmodel 是调用函数时传入的模型对象，
    # 可以是原始 PyTorch 模型，也可以是 torch2trt 转换后的 TensorRT 模型。

    correct = 0
    # correct：到目前为止预测正确的图片数量。

    total = 0
    # total：到目前为止已经测试的图片总数。

    with torch.no_grad():
        # 测试阶段不需要反向传播。
        # no_grad() 会关闭梯度记录，减少显存占用和额外计算。

        for batch_idx, (inputs, targets) in enumerate(testloader):
            # (语法) enumerate(testloader) 每次返回“batch 编号、batch 数据”。
            # batch 数据本身又包含 inputs 和 targets，所以这里使用嵌套拆包。
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
            # sum() 统计当前 batch 预测正确的数量；
            # item() 转为 Python 数值。

    acc = 100. * correct / total
    # 用百分数表示准确率，例如 83.05 表示 83.05%。

    return acc
    # return 把准确率返回给调用 test(...) 的代码。


if __name__ == "__main__":
    # __name__ 是 Python 自动设置的特殊变量：
    # - 直接执行 python int8_infer.py 时，__name__ 等于 "__main__"；
    # - 被其它文件 import 时，__name__ 等于模块名 "int8_infer"。
    # 所以这个 if 可以防止“只想导入函数时却自动开始 TensorRT 实验”。

    ################################################
    # 解析命令行参数                                 #
    ################################################
    parser = argparse.ArgumentParser(description='Test for Cifar10 w/ or w/o tensorrt')
    # ArgumentParser 是 argparse 中的类；这里调用它创建 parser 对象。
    # description 是关键字参数，它的内容会显示在 --help 帮助信息中。

    parser.add_argument('--gpu', '-p', action='store_true', help='Trained on GPU')
    # add_argument(...) 是 parser 对象提供的方法，用来声明一个命令行参数：
    # - --gpu 和 -p 是同一个参数的长写法和短写法；
    # - action='store_true' 表示写了该开关时 args.gpu=True，否则为 False；
    # - help 是执行 --help 时显示的说明。

    parser.add_argument('--model', '-m', default='alexnet', type=str, help='Name of Network')
    # 声明模型名称参数：
    # - --model 和 -m 是同一个参数的长写法和短写法, i.e., -m alexnet 和 --model alexnet 的作用相同
    # - default='alexnet'：没有指定参数时使用 AlexNet；
    # - type=str：把命令行输入保存为字符串；

    args = parser.parse_args()
    # parse_args() 真正读取终端参数，并把结果放进 args 对象。
    # 后面可以通过 args.gpu、args.model 访问用户传入的值。

    ################################################
    # 创建模型并加载训练好的权重                       #
    ################################################
    model_name = args.model
    # model_name 是变量，保存用户指定的模型名，例如字符串 'alexnet'。
    #
    model = model_factory(model_name)
    # model_factory(model_name) 是函数调用，根据字符串创建对应的网络对象。
    # 此时只建立了 AlexNet 的层和连接关系，还没有加载训练得到的参数。
    #
    print("Testing model: %s" % model_name)
    # %s 会被 model_name 的字符串内容替换，并把实际模型名打印到终端。

    if args.gpu and torch.cuda.is_available():
        # 当前 TensorRT 转换和推理依赖 NVIDIA CUDA GPU，所以必须同时满足：
        # 1. 用户在命令行中写了 -p/--gpu；
        # 2. PyTorch 检测到可用的 CUDA GPU。

        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn 是 PyTorch 的 CuDNN 配置对象。
        # enabled=True 表示允许使用 NVIDIA CuDNN 深度学习加速库。

        torch.backends.cudnn.benchmark = True
        # 让 CuDNN 针对固定的输入尺寸寻找较快的卷积实现。
        # 本实验输入始终是 1×3×32×32，适合开启 benchmark。

        model = model.cuda()
        # .cuda() 把模型参数和缓冲区从 CPU 内存复制到 GPU 显存。
        # 返回的 GPU 模型重新赋值给 model。
    else:
        print("-p is a must when executing this script. Exiting...")
        # 没有 -p 或没有 CUDA 时无法继续进行 TensorRT 推理。
        exit(); # 立即结束程序。
        # 末尾分号在 Python 中没有必要，但不会改变功能。

    checkpoint_path = EXP3_DIR / 'weights' / 'alexnet' / 'weights.130.83.050.pt'
    # exp2 保存 FP32 AlexNet 权重的格式是：
    # weights/alexnet/weights.<epoch>.<accuracy>.pt。
    # 本实验使用 epoch=130、accuracy=83.050 的 checkpoint。

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"AlexNet checkpoint not found: {checkpoint_path}\n"
            "Copy weights.130.83.050.pt into exp3/weights/alexnet/ before running exp3."
        )
        # 在构建 TensorRT engine 前明确检查权重，避免 torch.load() 给出难理解的路径错误。

    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    # map_location='cuda' 表示把 checkpoint 中的张量加载到 CUDA GPU。

    model.load_state_dict(checkpoint['net'])
    # 这一行从里向外执行：
    # 1. checkpoint['net'] 取出模型参数 state_dict；
    # 2. load_state_dict(...) 把这些参数装入刚创建的模型。

    accbefore = checkpoint['acc']
    # checkpoint['acc'] 是训练代码保存权重时记录的测试准确率。
    # 后面的完整测试集流程会重新计算 accuracy，并覆盖 accbefore 的当前值。

    ################################################
    # 定义模型输入所需的图像预处理                      #
    ################################################
    cali_augmentation = transforms.Compose([
        # cali_augmentation 是变量，保存组合后的预处理对象。
        # Compose 会让每张输入图片依次经过 Resize -> ToTensor -> Normalize。

        # transforms.RandomCrop(32, padding=4),
        # 随机裁剪适合训练时做数据增强，但校准和推理需要稳定、可重复的输入，
        # 所以这里将它注释掉。

        transforms.Resize((32, 32), interpolation=Image.BICUBIC),
        # 将输入图像缩放到 CIFAR-10 模型要求的 32×32 像素。
        # interpolation=Image.BICUBIC 表示缩放时使用双三次插值。

        transforms.ToTensor(),
        # 把 PIL 图片转换为 PyTorch 张量：形状变为 [3, 32, 32]，
        # 同时把像素值从 0～255 缩放到 0～1。

        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # 按 RGB 三个通道分别执行 (像素值 - mean) / std。
        # 这些均值和标准差与 CIFAR-10 训练时使用的预处理保持一致。
    ])

    cali_cifar10 = QDataset(transform=cali_augmentation)
    # QDataset(...) 表示调用 data.py 中 QDataset 类的构造方法，创建数据集对象。
    # transform=cali_augmentation 把上面的预处理流程传给它。
    #
    # cali_cifar10 会在下面通过 int8_calib_dataset 参数传给 torch2trt。
    # TensorRT 会让这些图片经过网络，并观察各层 activation values 的数值分布。

    ################################################
    # 将 PyTorch 模型转换成 TensorRT INT8 引擎        #
    ################################################

    model.eval()
    # eval() 把模型切换到评估/推理模式：
    # - BatchNorm 使用训练阶段保存的均值和方差；
    # - Dropout 停止随机丢弃神经元。
    # eval() 不等于关闭梯度；关闭梯度需要另外使用 torch.no_grad()。


    x = torch.randn([1, 3, 32, 32]).cuda()  # no .half()?
    # 创建示例输入，形状为 [1, 3, 32, 32]：
    # - 1：batch size，一次输入一张图片；
    # - 3：RGB 三个颜色通道；
    # - 32×32：图片尺寸。
    #
    # torch.randn 默认创建 FP32 张量，随后 .cuda() 把它移动到 GPU。
    #
    # 这里没有调用 .half()，所以示例输入仍是 FP32；TensorRT 根据转换参数决定各层精度。

    model_trt_int8 = torch2trt(model, [x],
                          fp16_mode=False, int8_mode=True, max_batch_size=1,
                          int8_calib_dataset=cali_cifar10,
            int8_calib_algorithm=tensorrt.CalibrationAlgoType.ENTROPY_CALIBRATION_2)
    # 把 PyTorch 模型转换成 TensorRT 模型：
    # - model：已经加载权重并切换到 eval 模式的 PyTorch 模型；
    # - [x]：用列表包住模型的一个示例输入，（模型示例输入必须放在列表中）
    #        torch2trt 会通过这个输入追踪计算图并确定各层张量形状
    # - fp16_mode=False：本实验不主动启用 FP16，重点比较 FP32 和 INT8；
    # - int8_mode=True：要求 TensorRT 为支持的层启用 INT8；
    # - max_batch_size=1：engine 一次最多处理一张图片；
    # - int8_calib_dataset=cali_cifar10：把校准图片交给 TensorRT；
    # - int8_calib_algorithm=...：使用熵校准算法选择 INT8 范围。
    #
    #
    # 【weights 在哪里量化？】
    # 代码中没有单独调用 quantize_weights(...)。
    # 设置 int8_mode=True 后，TensorRT builder 在构建 engine 时读取 model 中训练好的 FP32 weights，
    # 为支持 INT8 的层计算权重 scale，并在 engine 中保存/使用量化后的权重。
    # 权重数值本身已经确定，因此权重量化范围通常不需要校准图片。
    #
    # 【activation values 在哪里量化？】
    # int8_calib_dataset=cali_cifar10 让 TensorRT 在构建 engine 时使用校准图片，
    # 执行前向传播，收集各层输入和输出 activation values 的数值分布。
    #
    # ENTROPY_CALIBRATION_2 的核心可以按下面的顺序理解：
    # 1. 根据收集到的 activation values 建立数值分布/直方图；
    # 2. 尝试不同的截断阈值 T，把超出 [-T, T] 的极端值截断；
    # 3. 对截断后的分布模拟 INT8 量化，再与原始分布计算 KLD（Kullback-Leibler Divergence，相对熵）；
    # 4. 选择让量化前后分布差异（KLD）最小的 T，作为该张量的动态范围；
    # 5. 再由动态范围 T 推导 INT8 scale。以常见的对称有符号 INT8 为例，
    #    可以近似理解为 scale = T / 127，量化约为 q = round(x / scale)。
    #
    # 所以“根据分布选择 activation 的 INT8 scale”是一个压缩说法：
    # 先按照 KLD 原则选择截断阈值 T，再根据 T 计算浮点数与 INT8 之间的 scale。
    # 正式推理时，中间 activation values 会按照校准得到的动态范围/scale 使用 INT8 表示。

    # 注意：没有 INT8 实现的算子可能回退到其它支持的精度；
    # 所以启用 INT8 不代表 engine 中每一个算子一定都执行 INT8。

    ################################################
    # 读取一张图片并进行 FP32/INT8 推理对比            #
    ################################################
    img = Image.open('test.jpeg')
    # Image.open(...) 是函数调用，打开测试图片并返回 PIL Image 对象。
    # 这个对象既用于生成模型输入，也会在后面直接画上文字。

    img_tensor = cali_augmentation(img)
    # cali_augmentation(...) 表示像调用函数一样调用 Compose 对象，
    # 对图片依次执行缩放、转张量和标准化。
    #
    # 这里是在准备正式推理输入，不是在重新执行 INT8 校准。
    # 校准图片和正式测试图片必须使用相同的 Resize、ToTensor 和 Normalize。
    # 原因是：TensorRT 根据“预处理后的校准图片”统计各层 activation values 的范围；
    # 如果正式图片采用不同预处理，它进入模型后的数值分布也会改变，之前得到的
    # INT8 截断阈值 T 和 scale 就可能不再适合正式输入。

    # cali_augmentation(img) 当前只处理一张图片，因此得到 [C, H, W]，即 [3, 32, 32]；
    # 模型要求输入包含 batch 维度 [N, C, H, W]，
    # 所以下一行还要用 unsqueeze_(0) 在最前面添加 N=1。
    img_tensor = img_tensor.unsqueeze_(0).cuda()
    # 方法名末尾的下划线表示原地操作。unsqueeze_(0) 在最前面增加 batch 维度：
    # [3, 32, 32] -> [1, 3, 32, 32]，然后将张量复制到 GPU。

    warmup_runs = 50
    # FP32 和 INT8 各自先预热 50 次。预热可以让 CUDA、CuDNN 和 TensorRT
    # 完成初始化、算法选择和缓存建立；预热耗时不计入正式结果。

    measured_runs = 100
    # 每种模型正式推理 100 次，再用总时间除以 100，得到平均单次 latency。

    ################################################
    # FP32：先预热 50 次，再正式测量 100 次        #
    ################################################
    with torch.no_grad():
        # 推理不需要反向传播。no_grad() 关闭梯度记录，减少额外计算和显存占用。

        for _ in range(warmup_runs):
            y_fp32 = model(img_tensor)
            # 把同一个 img_tensor 重复送入 FP32 模型。
            # 循环次数变量不会在循环体中使用，所以按 Python 惯例命名为 _。

        torch.cuda.synchronize()
        # CUDA 默认异步执行：CPU 提交 GPU 任务后可以立即继续。
        # 正式计时前先等待 50 次预热全部完成，避免预热任务混入正式测量。

        start_time = time.perf_counter()
        # perf_counter() 是高精度单调计时器，比 time.time() 更适合测量短时间间隔。

        for _ in range(measured_runs):
            y_fp32 = model(img_tensor)
            # 正式执行 100 次 FP32 推理。
            # 循环结束后，y_fp32 保留最后一次输出，供后面 softmax 和图片标注使用。

        torch.cuda.synchronize()
        # 读取结束时间前等待 GPU 完成全部 100 次推理；否则可能只测到 CPU 提交任务的时间。

        total_time_fp32 = time.perf_counter() - start_time
        # 结束时间减去开始时间，得到 100 次 FP32 推理总时间，单位为秒。

    time_spent_fp32 = total_time_fp32 / measured_runs
    # 总时间除以正式测量次数，得到 FP32 平均单次推理耗时，单位仍是秒。

    print('Average Time Spent for fp32: {:.3f}ms'.format(time_spent_fp32 * 1000))
    # 乘以 1000 把秒转换为毫秒；{:.3f} 表示保留小数点后三位。

    percentage_fp32 = torch.softmax(y_fp32[0], dim=0) * 100
    # y_fp32[0] 取 batch 中第一张图片的十个 logits。
    # softmax(..., dim=0) 沿十个类别的维度计算概率，结果总和为 1；
    # 再乘 100，把 0～1 的概率转换为百分比。
    cl_fp32, index_fp32 = torch.max(percentage_fp32, 0)  # 概率最大值，模型认为的物体
    # torch.max(..., 0) 返回：
    # - cl_*：最高类别概率；
    # - index_*：最高概率对应的类别编号，即模型预测结果。

    ################################################
    # TensorRT INT8：先预热 50 次，再测量 100 次    #
    ################################################
    with torch.no_grad():
        for _ in range(warmup_runs):
            y_trt_int8 = model_trt_int8(img_tensor)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(measured_runs):
            y_trt_int8 = model_trt_int8(img_tensor)
            # 使用同一个 img_tensor 正式执行 100 次 INT8 推理。
            # y_trt_int8 最后保留第 100 次输出，供后面的预测结果计算使用。
        torch.cuda.synchronize()
        total_time_int8 = time.perf_counter() - start_time

    time_spent_int8 = total_time_int8 / measured_runs
    # INT8 总时间除以 100，得到平均单次推理耗时。

    print('Average Time Spent for int8: {:.3f}ms'.format(time_spent_int8 * 1000))
    percentage_trt_int8 = torch.softmax(y_trt_int8[0], dim=0) * 100
    cl_trt_int8, index_trt_int8 = torch.max(percentage_trt_int8, 0)

    ###########################################################
    # 开始画图                                                 #
    ###########################################################

    draw = ImageDraw.Draw(img) # 创建绑定到 img 的绘图对象。
    # 后续调用 draw.text(...) 会直接修改内存中的这张图片。

    classes = ['plane', 'car', 'bird', 'cat', 'deer', \
               'dog', 'frog', 'horse', 'ship', 'truck']
    # classes 是 Python 列表，
    # 下标 0～9 对应 CIFAR-10 的十个类别编号。

    try:
        font = ImageFont.truetype('LiberationSans-Regular.ttf', 30)
        # 优先从当前工作目录加载 LiberationSans-Regular.ttf，字号为 30。
        # 当前目录没有字体文件，或者字体无法读取时，truetype() 会抛出 OSError。
        # except 会捕获这个错误，改用 PIL 内置默认字体，让实验继续运行。
    except OSError:
        font = ImageFont.load_default()
        # 默认字体可能比较小，但不会影响模型推理、预测结果或 latency.

    text = 'Mode: fp32, ' + 'avg {:.2}ms '.format(time_spent_fp32 * 1000) \
        + str(classes[index_fp32.item()]) \
        + ' (' + '{:.2f}'.format(cl_fp32.item()) + '%' + ')'
    # 拼接 FP32 结果文字，包括推理耗时、预测类别和最高概率。
    # - index_fp32.item() 把预测下标张量转换成 Python 整数，再从 classes 取类别名；
    # - cl_fp32.item() 把只含一个值的张量转换成普通 Python 数值。
    # - {:.2} 是“两位有效数字”，不是“小数点后两位”；{:.2f} 才是两位小数。
    draw.text((0, 0), text, font=font, fill="#f000ff", spacing=0, align='left')
    # 在图片左上角 (x=0, y=0) 写入 FP32 结果；
    # fill 指定文字颜色。
    print(text + '\n')

    text = 'Mode: int8, ' + 'avg {:.2}ms '.format(time_spent_int8 * 1000) \
        + str(classes[index_trt_int8.item()]) \
        + ' (' + '{:.2f}'.format(cl_trt_int8.item()) + '%' + ')'
    # 用相同格式生成 TensorRT INT8 的耗时、类别和置信度文字。
    draw.text((0, 40), text, font=font, fill="#ff00ff", spacing=0, align='left')
    # 在 y=40 的位置写入 TensorRT INT8 结果，避免与第一行重叠。
    print(text)
    # 在终端输出 INT8 的推理结果。

    img.save(EXP3_DIR/'test_result.jpg')
    # 将已经画上两行文字的 img 保存为 test_result.jpg

    ###########################################################
    # 在完整测试集上分别测量原 FP32 模型、TensorRT INT8模型的       #
    # 准确率、总耗时                                            #
    ###########################################################
    # 注意：这里测量的是遍历完整测试集的“端到端总时间”，其中不仅有模型推理，
    # 还包含 DataLoader 读取/预处理图片、CPU 到 GPU 的数据复制和 Python 循环。
    # 前面重复同一个 img_tensor 得到的平均 latency 主要测 GPU 模型推理，
    # 两处计时包含的工作不同，因此不能把两个数值当成同一种指标直接比较。

    # FP32 模型在完整测试集上的准确率和总耗时
    torch.cuda.synchronize()
    start_time = time.perf_counter() # 得到的时间单位为秒
    accbefore = test(model)
    torch.cuda.synchronize()
    time_spent_fp32_test = time.perf_counter() - start_time
    print(
        "Acc for fp32: %.2f%%; Total time: %.3fms"
        % (accbefore, time_spent_fp32_test * 1000)
    )
    # accbefore 已经是百分数；格式字符串中的 %% 用于输出一个 % 符号。

    # TensorRT INT8 模型在完整测试集上的准确率和总耗时
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    accafter = test(model_trt_int8)
    torch.cuda.synchronize()
    time_spent_int8_test = time.perf_counter() - start_time
    print(
        "Acc for trt int8: %.2f%%; Total time: %.3fms"
        % (accafter, time_spent_int8_test * 1000)
    )
