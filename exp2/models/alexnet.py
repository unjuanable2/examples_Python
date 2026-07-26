import torch.nn as nn
# torch.nn 提供构建神经网络需要的层和基础类。
# 使用 as nn 起别名后，可以写 nn.Module、nn.Conv2d、nn.Linear 等简洁形式。


NUM_CLASSES = 10
# CIFAR-10 一共有 10 个类别，所以 AlexNet 最后一层输出 10 个 logits。
# logits 是模型对每个类别给出的原始分数，还不是概率。


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

            nn.MaxPool2d(kernel_size=2, stride=2),
            # kernel_size=2：每次在 2x2 区域中取一个最大值。
            # stride=2：池化窗口每次横向或纵向移动 2 个像素。
            # 因为窗口大小和移动步长都是 2，窗口之间不重叠，宽和高都会减半：16x16 -> 8x8。

            # 如果只写 nn.MaxPool2d(kernel_size=2)，PyTorch 默认也会令
            # stride=kernel_size，也就是 stride=2；

            # 第 2 个卷积块：
            # [N, 64, 8, 8] -> [N, 192, 8, 8] -> [N, 192, 4, 4]
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            # stride 没有写时默认是 1。
            # kernel_size=3、stride=1、padding=1 会保持特征图宽高不变。

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 同样使用 2x2 窗口和步长 2，使特征图从 8x8 变为 4x4。

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
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 同样使用 2x2 窗口和步长 2，使特征图从 4x4 变为 2x2。
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
