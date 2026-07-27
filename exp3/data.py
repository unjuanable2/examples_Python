import torch
# torch 是 PyTorch 的主模块。
#
# 这个文件使用 torch.utils.data.DataLoader，把 CIFAR-10 测试集包装成可以按 batch 读取的对象。

from PIL import Image
# PIL 是 Python 常用的图像处理库，Image 用于打开 data/q 目录中的 JPG 图片。
#
# (语法) from PIL import Image 表示只从 PIL 包中导入 Image，
# 后面可以直接写 Image.open(...)，不用写 PIL.Image.open(...)。

from torch.utils.data import Dataset
# Dataset 是 PyTorch 提供的数据集基类。

# 后面的 QDataset 会继承 Dataset，从而能够被 torch2trt/PyTorch 当作数据集使用。
#
# (语法) 类的继承写成 class QDataset(Dataset):
# 可以理解为：QDataset 在 Dataset 的基础上定义自己的取样方式和数据集长度。

from torchvision import datasets, transforms
# torchvision.datasets 提供 CIFAR-10 等常用数据集的读取接口。
# torchvision.transforms 提供 ToTensor、Normalize 等图像预处理操作。

from glob import glob
# glob 用来按照通配符查找文件。
# 例如 glob("./data/q/*.jpg") 会返回 data/q 目录下所有扩展名为 .jpg 的文件路径。

from pathlib import Path
# Path 用对象表示文件路径；下面用它根据 data.py 的位置找到 exp1/data。


################################################
# 定义 CIFAR-10 测试集的预处理流程                 #
################################################
test_transforms = transforms.Compose([
    # transforms.Compose([...]) 把列表中的多个预处理操作按顺序组合起来。
    # 每张 CIFAR-10 测试图片会依次经过 ToTensor -> Normalize。

    transforms.ToTensor(),
    # 把图片转换为 PyTorch 浮点张量：
    # - 通道顺序变为 [C, H, W]，CIFAR-10 RGB 图片通常是 [3, 32, 32]；
    # - 像素值从整数 [0, 255] 缩放为浮点数 [0, 1]。

    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # 对 R、G、B 三个通道分别执行：normalized = (value - mean) / std。
    # 第一个列表是三个通道的均值，第二个列表是三个通道的标准差。
    # 这些数值需要和模型训练时使用的预处理保持一致。
])


################################################
# 创建 CIFAR-10 测试集对象                       #
################################################

EXP3_DIR = Path(__file__).resolve().parent
# __file__ 是当前 data.py 的路径；resolve().parent 得到 exp3 目录。

EXP1_DATA_DIR = EXP3_DIR.parent / "exp1" / "data"
# EXP3_DIR.parent 是项目根目录，再拼接 exp1/data。
# 因此无论从哪个目录运行程序，CIFAR-10 都会读取仓库中的 exp1/data。


testset = datasets.CIFAR10( root=str(EXP1_DATA_DIR), 
    train=False, download=True, transform=test_transforms)
# datasets.CIFAR10(...) 创建一个 CIFAR-10 数据集对象，并把它保存在 testset 变量中。
# 括号中的参数含义：
# - root=str(EXP1_DATA_DIR)：读取或保存仓库 exp1/data 中的 CIFAR-10 数据；
# - train=False：选择官方测试集，而不是训练集；
# - download=True：目录中没有完整数据时自动尝试下载；
# - transform=test_transforms：每次取出图片时应用上面定义的测试预处理。


################################################
# 创建测试集 DataLoader                         #
################################################
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=4)
# DataLoader 把 testset 包装成一个可以按 batch 遍历的对象。
# - testset：上面创建的 CIFAR-10 测试集；
# - batch_size=1：每个 batch 只读取一张图片和一个标签；
# - shuffle=False：不打乱测试集顺序，使每次评估顺序固定；
# - num_workers=4：使用 4 个子进程并行读取和预处理图片。
#
# int8_infer.py 中的
#   for batch_idx, (inputs, targets) in enumerate(testloader):
# 会通过这个对象逐批取得图片 inputs 和真实标签 targets。


################################################
# 定义 INT8 量化校准图片数据集                 #
################################################
class QDataset(Dataset):
    # QDataset 是自己定义的数据集类，用于读取 ./data/q/*.jpg。
    # 它继承 Dataset，并实现 PyTorch 数据集最重要的三个部分：

    # 初始化数据集
    def __init__(self, transform):
        # __init__ 是构造方法。执行 QDataset(transform=...) 创建对象时会自动调用它。
        # transform 参数是外部传入的图像预处理流程。

        self.imagelist=glob("./data/q/*.jpg")
        # 查找当前工作目录下 data/q 中的全部 JPG，并把路径列表保存为对象属性 imagelist。
        # self 表示“当前这个 QDataset 对象”；
        # self.imagelist 可以被类中的其它方法访问。
        #                这里使用相对路径，所以通常需要先进入 exp3 目录再运行程序。

        self.len=len(self.imagelist)
        # len(...) 返回图片路径列表的元素数量，也就是校准数据集的样本数。
        # 结果保存在 self.len，供下面的 __len__() 返回。

        self.transform=transform
        # 保存外部传入的预处理对象。
        # 后面读取每张校准图片时，会调用同一个 transform 进行处理。

    # 规定按下标怎样取出一个样本
    def __getitem__(self, index):
        # 当程序执行 dataset[index] 时，Python 会自动调用这个方法。
        # index 是需要读取的样本下标，例如 0 表示第一张图片。

        image_path = self.imagelist[index]
        # 根据 index 从图片路径列表中取出当前图片路径。

        image = Image.open(image_path).convert("RGB")
        # 使用 PIL 打开图片，并统一转换成 RGB 三通道。

        input_tensor = self.transform(image).unsqueeze(0).cuda()
        # transform(image) 执行 Resize、ToTensor、Normalize，得到 [3, 32, 32]；
        # unsqueeze(0) 添加 batch 维度，得到 [1, 3, 32, 32]；
        # cuda() 把校准输入移动到 TensorRT 使用的 GPU。

        return [input_tensor]
        # torch2trt 会把列表中的每个张量对应到模型的一个输入。
        # AlexNet 只有图片这一个输入，所以返回 [input_tensor]。
        # INT8 校准只观察各层 activation values 的范围，不需要类别 label；
        # 如果返回 (input_tensor, label)，label 会被误认为模型的第二个输入。

    # 返回数据集包含多少个样本
    def __len__(self): # 当程序执行 len(dataset) 时，Python 会自动调用这个方法。
        return self.len
        # 返回初始化时统计到的 JPG 图片数量。
