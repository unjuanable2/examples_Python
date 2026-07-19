import torch
# torch.utils.data.DataLoader 用于实现数据集加载
#
# (语法) import torch：
# - import 表示导入模块；torch 是模块名；
# - 导入之后，就可以用 torch.xxx 的方式访问 torch 模块里的内容。

from torchvision import datasets, transforms
# torchvision.datasets 是 pytorch 中的一个数据集模块，提供了常用的数据集加载接口，
#   如 MNIST、CIFAR-10、ImageNet 等。
#
# torchvision.transforms 是 pytorch 中的一个图像预处理模块，提供了常用的图像变换操作，如裁剪、旋转、翻转、归一化等。
#   一般用 Compose 将多个变换组合在一起，形成一个完整的图像预处理流程。
#
# (语法) from torchvision import datasets, transforms：
# - torchvision 是一个包；
# - datasets 和 transforms 是 torchvision 里面的两个模块/对象；
# - 这样导入后，后面可以直接写 datasets.CIFAR10、transforms.Compose，
#   不需要每次都写 torchvision.datasets.CIFAR10。


################################################
# 定义训练集的数据增强和预处理流程                 #
################################################
train_transforms = transforms.Compose([
    # train_transforms 是变量，保存一个 Compose 对象。

    # transforms.Compose([...]) 是函数/类调用，作用是把列表里的多个图像变换按顺序串起来。
    # (语法) [...] 表示列表 list。
    # 列表里每一个元素都是一个 transform 对象, i.e., CIFAR-10 的每一张训练图片被读取出来后，
    # 会依次经过下面这些变换：RandomCrop -> RandomHorizontalFlip -> ToTensor -> Normalize

    transforms.RandomCrop(32, padding=4),
    # 作用：
    # 对图像进行随机裁剪，裁剪出一个大小为 32x32 的区域，
    # 并 (padding=4 是关键字参数) 在裁剪前对图像进行 4 像素的填充。
    #
    # 目的：增加数据的多样性，提高模型的泛化能力。
    # 
    # 理解: 
    # CIFAR-10 原图本来就是 32x32。如果直接裁剪 32x32，其实没有变化。
    # 先 padding=4 后，图像会先变成更大的区域，再随机裁回 32x32。
    # 这样同一张图片里的物体会产生轻微平移。
    #
    # 如果不做这个操作：
    #   模型仍然可以训练，但它更容易记住训练图片里物体固定出现的位置，
    #   测试图片中物体位置稍有变化时，泛化能力可能变差。

    transforms.RandomHorizontalFlip(),
    # 作用：
    # RandomHorizontalFlip() 后面有括号，表示创建一个随机水平翻转的变换对象。
    # 这里没有显式写 p=0.5，因为它的默认概率就是 0.5,
    # i.e., 每张训练图片有 50% 的概率被水平翻转。
    #
    # 理解: 
    # CIFAR-10 里的很多类别，比如 car、cat、dog、ship，左右翻转后通常仍然属于同一类。
    # 这等价于给训练集增加了更多“看起来不完全一样，但标签不变”的样本。
    #
    # 如果任务是识别数字、文字、方向箭头等，水平翻转可能改变语义，
    # 那就不能随便使用这个增强。

    transforms.ToTensor(),
    # 作用：
    # 将 numpy 格式的图像数据转换为 PyTorch 的 FloatTensor (张量) 格式，
    # 把通道顺序变成 PyTorch 常用的 [C, H, W] ( i.e., [颜色通道数, 高度, 宽度] )，
    # 将像素值 从 [0, 255] 归一化到 [0, 1] 范围内。
    #
    # 理解: 神经网络不能直接吃 PIL 图片对象，它需要张量 Tensor。
    # CIFAR-10 是 RGB 图片，所以一张图经过 ToTensor 后形状通常是 [3, 32, 32]。

    transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    # 作用：对图像进行归一化处理，将 RGB 每个通道的像素值 - 均值 并除以标准差，
    # i.e., 对每个像素通道做 x_normalized = (x - mean) / std
    # - [0.4914, 0.4822, 0.4465]：训练数据中 R、G、B 三个通道各自的均值
    # - [0.2023, 0.1994, 0.2010]：训练数据中 R、G、B 三个通道各自的标准差 
    # 这里的 mean/std 是 CIFAR-10 数据集常用统计值。
    #
    # 目的：使得图像的像素值分布更加集中，有助于模型的训练和收敛。
    #
    # 为什么 Normalize 要放在 ToTensor 后面 ?
    # - Normalize 处理的是 Tensor；
    # - ToTensor 之前还是图像对象，不适合直接做这个张量归一化。
])


################################################
# 定义测试集的预处理流程                           #
################################################
# 测试集的图像预处理流程
test_transforms = transforms.Compose([
    # test_transforms 是一个 Compose 对象。

    # 测试集只做确定性的预处理，不做随机增强。
    # 原因是：测试/评估时希望同一张图片每次输入模型都完全一样，这样准确率才稳定、可比较。

    transforms.ToTensor(),

    transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    # 测试集也要使用和训练集相同的 Normalize。
    # 原因：模型训练时看到的是归一化后的输入；
    # 测试时也必须用同样的数据分布，否则模型会“看见”和训练时不同尺度的数据。
])


##################################################
# 创建 CIFAR-10 数据集对象                         #
##################################################
trainset = datasets.CIFAR10( root='./data', train=True, download=True, transform=train_transforms)
# trainset 是变量，保存 CIFAR-10 训练集对象。
# 更像一个“可以按下标取样本的数据集合”：
# - trainset[0] 可以取出第 0 张图片和标签；
# - len(trainset) 可以得到训练集大小。

# datasets.CIFAR10(...) 表示创建一个 CIFAR10 数据集对象。
# 这里的 root、train、download、transform 都是关键字参数：
# - root='./data'：CIFAR-10 数据集保存/读取的根目录
# - train=True：读取的是 CIFAR-10 官方训练集，大小是 50000 张；
# - download=True：如果 root 下面没有数据，就尝试自动下载；
# - transform=train_transforms：每次取出训练图片时，应用上面定义的训练预处理流程。

testset = datasets.CIFAR10( root='./data', train=False, download=True, transform=test_transforms)
# testset 是变量，保存 CIFAR-10 测试集对象。

# - train=False：读取的是 CIFAR-10 官方测试集，大小是 10000 张。
# - transform=test_transforms 表示测试图片只做 ToTensor 和 Normalize。

###################################################
# 创建 DataLoader：把数据集包装成按 batch 读取的对象    #
###################################################
trainloader = torch.utils.data.DataLoader( trainset, 
    batch_size = 128, shuffle=True, num_workers=4 )
    # - batch_size：批尺寸，默认为 1，=128 表示每次从训练集中取 128 张图片和 128 个标签
    # - shuffle：是否在每个 epoch 开始时随机打乱数据，默认为 False。
    #            =True：每个 epoch 打乱样本顺序，可以避免模型总是按固定顺序学习数据。
    # - num_workers：用几个子进程并行读取/预处理数据。

# trainloader 是变量，保存 DataLoader 对象。

# torch.utils.data.DataLoader(...) 的作用是把 trainset 包装成“可以一批一批吐出数据”的对象。
# (语法)：
# - torch 是模块；
# - utils 是 torch 里的子模块/命名空间；
# - data 是 utils 里的子模块；
# - DataLoader 是 data 里的类；
# - 调用 DataLoader(...) 会创建一个 DataLoader 对象。

# 理解:
# 在 train.py 里会看到类似：
#   for idx, (inputs, targets) in enumerate(trainloader):
# 其中 inputs 就是一批图片，targets 就是一批标签。

testloader = torch.utils.data.DataLoader( testset, 
    batch_size = 128, shuffle=False, num_workers=4 )
# testloader 是变量，保存测试集的 DataLoader 对象。
# 它每次从 testset 里取 128 张测试图片和标签。
# shuffle=False 表示测试时不打乱测试集顺序。
# - 测试集只用来评估，不参与训练；
# - 是否打乱通常不会改变最终准确率；
# - 但固定顺序更方便复现和对比日志，所以测试集一般设成 False。
