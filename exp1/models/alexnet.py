import torch
# torch 是 PyTorch 的主模块。
# 这里主要使用 torch.softmax()、torch.max() 以及张量相关操作。

import torch.nn as nn
# torch.nn 提供构建神经网络需要的层和基础类。
# 使用 as nn 起别名后，可以写 nn.Module、nn.Conv2d、nn.Linear 等简洁形式。

import os
# os.path.join() 用来拼接权重文件和测试图片的路径。

from torchvision import transforms
# transforms 用来把测试图片缩放、转换成张量并进行标准化。

from PIL import Image, ImageDraw, ImageFont, ImageFilter
# PIL 用于读取、绘制和保存单张测试图片。
# ImageFilter 当前没有实际使用，保留它不会影响模型训练和推理。


NUM_CLASSES = 10
# CIFAR-10 一共有 10 个类别，所以 AlexNet 最后一层输出 10 个 logits。
# logits 是模型对每个类别给出的原始分数，还不是概率。


tran = transforms.Compose([
    # Compose 会让测试图片依次经过下面三个预处理步骤。

    transforms.Resize((32, 32), interpolation=Image.BICUBIC),
    # 把输入图片统一缩放为 32x32，以匹配当前 AlexNet 的输入尺寸。
    # BICUBIC 表示使用双三次插值完成缩放。

    transforms.ToTensor(),
    # 把 PIL 图片转换成形状为 [C, H, W] 的 FloatTensor，
    # 同时把像素值从 [0, 255] 缩放到 [0, 1]。

    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.201]
    )
    # 使用 CIFAR-10 的 RGB 均值和标准差进行标准化。
    # 推理时必须使用与训练时相同的标准化方法。
])


class AlexNet(nn.Module):
    # 定义 AlexNet 类，并继承 nn.Module。
    # 继承 nn.Module 后，PyTorch 才能自动管理这个模型中的层、参数和梯度，
    # 并允许使用 model.cuda()、model.train()、model.eval() 等方法。

    def __init__(self, num_classes=NUM_CLASSES):
        # __init__ 是构造方法，在执行 AlexNet() 创建模型对象时自动调用。
        # self 表示当前正在创建的 AlexNet 对象。
        # num_classes 表示分类类别数，默认使用 CIFAR-10 的 10 类。

        super(AlexNet, self).__init__()
        # 初始化父类 nn.Module。
        # 只有先初始化 nn.Module，PyTorch 才能正确登记下面定义的网络层。

        ############################################################
        # 特征提取部分：输入 [N, 3, 32, 32]，输出 [N, 256, 2, 2] #
        ############################################################
        self.features = nn.Sequential(
            # nn.Sequential 会按照写入顺序依次执行其中的网络层。
            # N 表示 batch size，即一次送入模型的图片数量。

            # 第 1 个卷积块：
            # [N, 3, 32, 32] -> [N, 64, 16, 16] -> [N, 64, 8, 8]
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            # in_channels=3：输入是 RGB 图片，有 3 个通道。
            # out_channels=64：使用 64 个卷积核，输出 64 张特征图。
            # kernel_size=3：每个卷积核的大小是 3x3。
            # stride=2：卷积核每次移动 2 个像素，所以宽高大约减半。
            # padding=1：在四周补 1 个像素，控制卷积后的特征图尺寸。

            nn.ReLU(inplace=True),
            # ReLU 执行 max(0, x)，给网络加入非线性表达能力。
            # inplace=True 表示尽量直接修改原张量，从而减少额外显存占用。

            nn.MaxPool2d(kernel_size=2),
            # 2x2 最大池化会从每个区域保留最大值，使宽和高再次减半。

            # 第 2 个卷积块：
            # [N, 64, 8, 8] -> [N, 192, 8, 8] -> [N, 192, 4, 4]
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            # stride 没有写时默认是 1。
            # kernel_size=3、stride=1、padding=1 会保持特征图宽高不变。

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 第 3 个卷积块：
            # [N, 192, 4, 4] -> [N, 384, 4, 4]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第 4 个卷积块：
            # [N, 384, 4, 4] -> [N, 256, 4, 4]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第 5 个卷积块：
            # [N, 256, 4, 4] -> [N, 256, 4, 4] -> [N, 256, 2, 2]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        ############################################################
        # 分类部分：把卷积特征转换成 10 个类别的 logits              #
        ############################################################
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # Dropout 在训练时随机把一部分特征设为 0，默认概率 p=0.5。
            # 它可以降低神经元之间的过度依赖，减少过拟合。
            # model.eval() 后 Dropout 会自动关闭随机丢弃行为。

            nn.Linear(256 * 2 * 2, 4096),
            # 卷积部分输出 [N, 256, 2, 2]。
            # 每张图片展开后有 256*2*2=1024 个特征，映射到 4096 个特征。

            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            # 第二个全连接层保持 4096 维。

            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
            # 最后一层输出 num_classes 个 logits。
            # CIFAR-10 中输出形状为 [N, 10]，每列对应一个类别。
            # 训练时 nn.CrossEntropyLoss 会直接接收 logits，
            # 所以模型的 forward 中不需要提前执行 softmax。
        )

    def forward(self, x):
        # forward 定义输入数据进入模型后依次进行哪些计算。
        # 当外部执行 outputs = model(inputs) 时，nn.Module 会自动调用此方法。
        # 对 CIFAR-10，x 的形状通常是 [batch_size, 3, 32, 32]。

        x = self.features(x)
        # 依次通过 5 个卷积层和 3 个池化层。
        # 输出形状由 [N, 3, 32, 32] 变成 [N, 256, 2, 2]。

        x = x.view(x.size(0), 256 * 2 * 2)
        # 全连接层要求输入是二维张量 [样本数, 特征数]。
        # x.size(0) 是当前 batch 的样本数 N，不能把它写死成 128，
        # 因为最后一个 batch 的样本数可能小于设定的 batch_size。
        # 展开后的形状是 [N, 1024]。

        x = self.classifier(x)
        # 通过全连接分类器，得到形状为 [N, num_classes] 的 logits。

        return x
        # 把 logits 返回给训练代码。
        # train.py 会使用它计算交叉熵损失和预测类别。


def test():
    # test() 用于加载已经训练好的 AlexNet 权重，对单张 test.jpg 做推理。
    # 它不是 main.py 中的训练流程；只有直接运行本文件时才会执行。

    net = AlexNet().cuda()
    # 创建 AlexNet 模型，并把模型参数移动到默认 CUDA GPU。
    # 因此直接运行此测试函数时，机器必须能够使用 CUDA。

    model_path = os.path.join("weights", "alexnet.pt")
    # 拼出权重路径 weights/alexnet.pt。

    print("Model PATH: " + model_path)

    checkpoint = torch.load(model_path)
    # 读取保存的 checkpoint 字典。

    net.load_state_dict(checkpoint['net'])
    # checkpoint['net'] 保存模型的参数，把它加载进刚创建的 AlexNet。

    test_image = os.path.join('test.jpg')
    img = Image.open(test_image)
    # 打开当前目录下的测试图片。

    img_tensor = tran(img)
    # 应用前面定义的预处理。
    # 此时 img_tensor 的形状是 [C, H, W]，即 [3, 32, 32]。

    input_tensor = img_tensor.unsqueeze_(0).cuda()
    # unsqueeze_(0) 在最前面增加 batch 维度：
    # [3, 32, 32] -> [1, 3, 32, 32]，然后把输入移动到 GPU。

    y = net(input_tensor)
    # 前向传播，得到形状为 [1, 10] 的 logits。

    percentage = torch.softmax(y[0], dim=0) * 100
    # 对第 0 张图片的 10 个 logits 做 softmax，将其转换成概率百分比。

    print('cat percentage:')
    print(percentage)

    cl_fp32, index_fp32 = torch.max(percentage, 0)
    # 找出最大概率 cl_fp32 以及对应的类别编号 index_fp32。

    classes = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    # CIFAR-10 标签编号 0~9 对应的英文类别名。

    font = ImageFont.truetype('LiberationSans-Regular.ttf', 30)
    # 加载字体，用于把预测结果绘制到图片上。

    draw = ImageDraw.Draw(img)
    # 创建一个可以在原图片上绘制文字的对象。

    text = (
        str(classes[index_fp32])
        + ' ('
        + '{:.2f}'.format(cl_fp32.item())
        + '%)'
    )
    # 组合预测类别和概率，例如 cat (87.35%)。

    draw.text(
        (0, 0), text, font=font, fill="#ff00ff", spacing=0, align='left'
    )
    # 在图片左上角用紫色文字写出预测结果。

    img.save(test_image, 'jpeg')
    # 把绘制后的图片覆盖保存回 test.jpg。


if __name__ == '__main__':
    # 只有执行 python models/alexnet.py 时条件才成立。
    # 当 model_factory_dict.py 导入 AlexNet 时，不会自动运行 test()。
    test()
