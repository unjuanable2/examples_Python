from __future__ import print_function

import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from utils import progress_bar


class Trainer(object):
    # (语法) class Trainer(object): 
    # 表示定义一个类，类名叫 Trainer。
    # object 是 Python 里所有类的基础父类之一；这里可以先理解成历史写法。
    #
    # 这个类的作用：
    # - 保存训练需要的状态，例如模型、学习率、是否使用 GPU、最好准确率等；
    # - 提供训练、测试、保存模型、加载模型等方法。

    def __init__(self, model_name, model, lr, train_on_gpu=False, fp16=False, loss_scaling=False):
        # (语法) 
        # __init__ 是一个特殊方法，通常叫“构造方法”或“初始化方法”。
        # - 当 main.py 里执行 Trainer(...) 创建 Trainer 对象时，Python 会自动调用这个 __init__ 方法。
        # - self 表示“当前这个正在被创建/使用的 Trainer 对象本身”；
        #   - 类里的方法第一个参数通常都写 self
        # - 参数含义：
        #   - model_name：字符串，模型名字，例如 'resnet18'
        #   - model：已经创建好的神经网络模型对象
        #   - lr：学习率
        #   - train_on_gpu：是否使用 GPU，默认 False
        #   - fp16：是否使用 FP16 半精度训练，默认 False
        #   - loss_scaling：是否使用 loss scaling，默认 False
        #
        # 这些参数来自 main.py 里的：
        #   trainer = Trainer(model_name, model, args.lr, train_on_gpu, args.fp16, args.loss_scaling)
        
        ################################# 初始化用户传进来的参数 ################################
        self.model_name = model_name
        # 保存模型名字，后面保存权重文件时会用到。

        self.model = model
        # 这里把外面传进来的 model 参数保存到对象内部，
        # 后面 train()、evaluate() 等方法就可以通过 self.model 使用这个模型。

        self.lr = lr
        # 保存学习率。

        self.train_on_gpu = train_on_gpu
        # 保存是否使用 GPU 的设置。

        if train_on_gpu and torch.backends.cudnn.enabled:
            # 如果用户要求使用 GPU，并且 CuDNN 已经启用，
            # 才允许根据用户传进来的 fp16 参数决定是否打开半精度训练。
            self.fp16_mode = fp16
            # self.fp16_mode 表示这个 Trainer 对象最终是否真的使用 FP16 模式。

            self.loss_scaling = loss_scaling
            # 保存是否启用 loss scaling。
            # loss scaling 常用于 FP16 训练，用来减少梯度太小导致的数值下溢问题。
        else:
            self.fp16_mode = False
            # 如果没有 GPU 或 CuDNN 不可用，就强制关闭 FP16 模式。

            self.loss_scaling = False
            # FP16 都关闭了，loss scaling 也就没有必要开启。

            print("CuDNN backend not available. Can't train with FP16.")
            # 打印提示：CuDNN 后端不可用，不能使用 FP16 训练。

        ############################ 初始化其它参数 ################################
        self.best_acc = 0
        # 记录目前为止最好的测试准确率。
        # 初始值为 0，后面 evaluate() 中如果准确率更高，就会更新它。

        self.best_epoch = 0
        # 记录最好准确率出现在哪一个 epoch。

        self._LOSS_SCALE = 128.0
        # loss scaling 的缩放倍数 (i.e., 如果开启 loss_scaling，就把 loss 放大 128 倍)
        # 128.0 = 2**7，用来在 FP16 训练中先放大 loss，
        # 反向传播后再把梯度按比例缩回来。
        #
        # (语法) (命名习惯) _LOSS_SCALE 前面有一个下划线，
        # 表示这是类内部使用的属性，不太希望外部代码直接改它。
        #
        # 理解：
        # FP16 能表示的小数范围比 FP32 小。
        # 如果梯度太小，FP16 可能把它直接表示成 0 (数值下溢)，导致模型学不到这部分信息。
        #
        # 所以 FP16 训练时常用做法是：
        # 1. 先把 loss 乘以一个较大的数，例如 128；
        # 2. loss.backward() 时，梯度也会跟着被放大；
        # 3. 真正更新参数前，再把梯度除以 128 缩回来。

        if self.train_on_gpu: # 如果最终决定使用 GPU，就把模型移动到 GPU 显存上。
            self.model = self.model.cuda()
            # .cuda() 是 PyTorch 模型对象的方法。
            # 它会把模型参数从 CPU 内存移动到 CUDA GPU 显存。
            # 后面训练时，输入数据也必须移动到 GPU，否则模型和数据不在同一个设备上会报错。

        if self.fp16_mode: # 如果最终启用了 FP16 模式，就把模型转换成半精度训练形式。
            self.model = self.network_to_half(self.model)
            # network_to_half(...) 是当前 Trainer 类里定义的方法。
            # 它会把模型大部分参数转成 FP16，同时让 BatchNorm 保持 FP32，
            # 因为 BatchNorm 用 FP32 更稳定。

            self.model_params, self.master_params = self.prep_param_list(self.model)
            # prep_param_list(...) 会准备两套参数：
            # - self.model_params：模型里真正用于 forward/backward 的 FP16 参数
            # - self.master_params：和上面参数数值对应的一份 FP32 拷贝
            #
            # 为什么 FP16 模式下要有两套参数？
            # - FP16 参数：放在模型里参与前向传播和反向传播，速度快、省显存。
            #   - 如果直接用 FP16 参数做 optimizer.step()：
            #     梯度更新量可能很小，FP16 表示不出来，可能被四舍五入成 0，导致参数几乎没变
            # - FP32 主参数：专门给优化器更新，数值范围更大、精度更高。
            #
            # 所以这份代码采用的思路是：
            # 1. 用 FP16 模型参数快速算 forward/backward；
            # 2. 把 FP16 参数上算出来的梯度复制到 FP32 主参数；
            # 3. 优化器只更新 FP32 主参数；
            # 4. 再把更新后的 FP32 主参数复制回 FP16 模型参数。
        

        ############################# 声明优化器 #############################
        
        # 可以把训练理解成三步：
        # 1. forward：模型根据输入图片算出预测结果；
        # 2. backward：根据 loss 算出每个参数应该往哪个方向改，也就是梯度；
        # 3. optimizer.step()：根据 loss.backward() 算出来的梯度，优化器修改模型参数，让下一次 loss 尽量变小。
        #
        # 所以优化器不是模型结构的一部分。
        # ResNet18 决定“怎么算预测”，SGD 优化器决定“参数怎么更新”。

        if not hasattr(self, 'optimizer'):
            # hasattr(self, 'optimizer') 用来检查当前对象是否已经有 optimizer 这个属性。
            # not 表示取反。意思是：如果当前 Trainer 对象还没有优化器，就创建一个。

            if self.fp16_mode:
                # FP16 模式下，优化器不直接更新 self.model_params。
                # 它更新的是 FP32 主参数 self.master_params。
                #
                # 后面 train() 里会看到：
                # 1. self.model_grads_to_master_grads(...)
                #    把 FP16 模型参数上的梯度复制到 FP32 主参数上；
                # 2. self.optimizer.step()
                #    用优化器更新 FP32 主参数；
                # 3. self.master_params_to_model_params(...)
                #    把更新后的 FP32 主参数复制回 FP16 模型参数。

                self.optimizer = optim.SGD(self.master_params, self.lr, momentum=0.9, weight_decay=5e-4)
                # optim.SGD(...) 创建一个随机梯度下降优化器对象。
                # - self.master_params：要被优化器更新的参数
                # - self.lr：学习率
                # - momentum=0.9：动量，让更新方向更平滑
                # - weight_decay=5e-4：权重衰减，用来减少过拟合
                #
                # SGD 的基本直觉：参数新值 = 参数旧值 - 学习率 * 梯度
                # momentum 和 weight_decay 是在这个基本更新规则上的改进。
            else: # 普通 FP32 模式下，模型参数本身就是 FP32。
                # 这时不需要额外的 master_params，优化器直接更新 self.model.parameters() 就可以。
                self.optimizer = optim.SGD(self.model.parameters(),
                    self.lr, momentum=0.9, weight_decay=5e-4)

        # self.scheduler 是学习率调度器对象。
        self.scheduler = MultiStepLR(self.optimizer, milestones=[10, 20, 50, 100, 180], gamma=0.1)
        # MultiStepLR 的作用：训练到指定 epoch 时，把学习率乘上 gamma。
        #
        # 参数含义：
        # - self.optimizer：要被调度学习率的优化器
        # - milestones=[10, 20, 50, 100, 180] (一个列表)：到这些 epoch 时调整学习率
        # - gamma=0.1：每次调整时，学习率变成原来的 0.1 倍

        # 如果要使用多张 GPU，可以用 DataParallel 包装模型。
        # if self.train_on_gpu:
        #     self.model = nn.DataParallel(self.model)

        print('\n Model: {} | Training on GPU: {} | Mixed Precision: {} | Loss Scaling: {}'
              .format(self.model_name, self.train_on_gpu, self.fp16_mode, self.loss_scaling))
        # 打印当前 Trainer 的关键配置。
        # .format(...) 是字符串格式化方法，会把括号里的变量填进前面的 {} 里。

    def prep_param_list(self, model):
        """
        为 FP16 训练准备两套参数。

        第一套: model_params
        - 来自 model.parameters()
        - 是模型里真正参与前向传播和反向传播的参数
        - 在 FP16 模式下，它们大部分是半精度参数

        第二套: master_params
        - 是从 model_params 复制出来的一份 FP32 参数
        - 不直接参与模型 forward, 专门交给 optimizer 更新

        为什么不只保留 FP16 参数 ？
        - FP16 计算快、省显存；
        - 但 FP16 精度低，参数更新量很小时可能表示不出来；
        - 所以训练时用 FP16 做计算，用 FP32 保存“更精细的权重版本”。

        后续训练循环里的关系是：
        FP16 model_params 算梯度
            -> 梯度复制到 FP32 master_params
            -> optimizer 更新 FP32 master_params
            -> FP32 master_params 复制回 FP16 model_params
        """
        model_params = [p for p in model.parameters() if p.requires_grad]
        # model_params 是一个列表，里面放的是模型中需要训练的参数。
        # p.requires_grad 为 True 表示这个参数需要计算梯度、需要被训练更新。

        master_params = [p.detach().clone().float() for p in model_params]
        # master_params 也是一个列表, 是从 model_params 复制出来的一份 FP32 参数。
        #
        # p.detach()：从原计算图里分离出来，避免和原参数共享梯度历史
        # clone()：复制一份新的数据
        # float()：转成 FP32

        for p in master_params:
            p.requires_grad = True
            # master_params 后面要被优化器更新，所以也需要梯度。

        # 返回两套参数。
        # 调用处用：
        # self.model_params, self.master_params = self.prep_param_list(self.model)
        # 分别接住这两个返回值。
        return model_params, master_params

    def master_params_to_model_params(self, model_params, master_params):
        """
        把 FP32 主参数复制回 FP16 模型参数。
        """
        # 这个函数只在 FP16 训练模式下使用。
        # 参数含义：
        # - model_params：模型里实际用于 forward/backward 的 FP16 参数列表
        # - master_params：优化器实际更新的 FP32 主参数列表
        #
        # 为什么需要这个函数？
        # FP16 模式下，optimizer.step() 更新的是 master_params。
        # 但是下一次 forward 时，模型用的是 model_params。
        # 所以每次更新完 master_params 后，都要把新数值复制回 model_params。

        for model, master in zip(model_params, master_params):
            # (语法) zip(model_params, master_params)
            # 会把两个列表按位置配对。
            #
            # 例如：
            # model_params = [m1, m2, m3]
            # master_params = [p1, p2, p3]
            # zip(...) 之后每次循环得到：(m1, p1), (m2, p2), (m3, p3)
            #
            # 这里的 model 和 master 不是“模型对象”和“主模型”，
            # 而是两套参数列表中位置对应的两个参数张量。

            model.data.copy_(master.data)
            # copy_(...) 是原地复制，把更新后的 FP32 主参数数值复制到对应的 FP16 模型参数里
            # master.data 是 FP32 主参数的数值；
            # model.data 是 FP16 模型参数的数值。
            #
            # 下划线结尾的 copy_ 是 PyTorch 命名习惯：
            # 表示这个操作会直接修改调用它的对象本身。

    def model_grads_to_master_grads(self, model_params, master_params):
        # 这个函数也只在 FP16 训练模式下使用。
        #
        # 作用：
        # 把 FP16 模型参数上的梯度，复制到 FP32 主参数上。
        #
        # 为什么要复制梯度？
        # - forward/backward 是用 FP16 模型参数算的，
        #   所以 loss.backward() 后，梯度先出现在 model_params 上；
        # - 但 optimizer 管的是 master_params；
        # - 因此 optimizer.step() 前，必须把梯度复制到 master_params.grad。

        for model, master in zip(model_params, master_params):
            # 同样是把两套参数列表中位置对应的参数配对。

            if master.grad is None:
                # master.grad 是这个 FP32 主参数对应的梯度。
                # 如果它现在还是 None，说明还没有给它分配梯度存储空间。
                master.grad = Variable(master.data.new(*master.data.size()))
                # 这里创建一个和 master.data 形状相同的新张量，用来存放梯度。
                # - master.data.size() 得到参数张量的形状；
                # - *master.data.size() 是把形状里的各个维度展开成参数传进去；
                # - Variable 是旧版本 PyTorch 里包装张量的写法。
                #   新版本 PyTorch 里 Variable 已经基本和 Tensor 合并了，但这份代码保留了老写法。

            master.grad.data.copy_(model.grad.data)
            # 把 FP16 模型参数上的梯度 model.grad.data，
            # 复制到 FP32 主参数的梯度 master.grad.data。
            #
            # 复制完后，optimizer.step() 才知道应该怎样更新 master_params。

    def BN_convert_float(self, module):
        '''
        配合 network_to_half 使用。
        BatchNorm 层需要保持单精度参数。
        这里递归查找所有子层，并把 BatchNorm 层转回 float。
        不能直接使用内置的 .apply，因为 .apply 会把函数应用到所有模块、参数和缓冲区，
        这样就无法根据模块类型只保护 BatchNorm 的 float 转换。
        '''
        # 参数含义：
        # - module：一个 PyTorch 模块，可以是整个模型，也可以是模型里的某一层。
        #
        # 这个函数会递归遍历 module 的所有子模块。
        # 如果某个子模块是 BatchNorm，就把它转回 FP32。
        #
        # 为什么 BatchNorm 不适合直接 FP16？
        # BatchNorm 会计算均值、方差这类统计量。
        # 这些统计量对数值精度更敏感，用 FP32 通常更稳定。

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            # (语法) isinstance(x, 类型)
            # 用来判断 x 是不是某个类型的对象。
            #
            # torch.nn.modules.batchnorm._BatchNorm 是 PyTorch 中 BatchNorm 系列层的基础类型。
            # BatchNorm1d、BatchNorm2d、BatchNorm3d 都属于这一类。

            module.float()
            # 如果当前模块是 BatchNorm，就把它转成 FP32。

        for child in module.children():
            # module.children() 会返回当前模块的直接子模块。
            # 例如一个 ResNet 里有很多 layer，layer 里又有 conv/bn/relu。

            self.BN_convert_float(child)
            # 递归调用自己，继续检查子模块的子模块。

        # 返回处理后的 module。
        return module

    class tofp16(nn.Module):
        """
        添加一个层，用来把输入转换成 FP16。
        这是一个模型包装层，实现的逻辑相当于：
            def forward(self, input):
                return input.half()
        """

        def __init__(self):
            # tofp16 也是一个 PyTorch 模块类。
            # 它继承自 nn.Module，所以也需要初始化父类。

            super(Trainer.tofp16, self).__init__()
            # (语法) super(...).__init__()
            # 表示调用父类 nn.Module 的初始化方法。
            # 这样 PyTorch 才能正确把这个类当作一个神经网络层来管理。

        def forward(self, input):
            # forward(...) 定义这个模块前向传播时做什么。
            # 参数 input 是进入这个层的输入张量。

            # half() 把输入张量转换成 FP16。
            # 这个小模块的唯一作用就是：
            # 让进入半精度模型的输入也变成半精度。
            return input.half()

    def network_to_half(self, network):
        """
        用对 BatchNorm 安全的方式把模型转换成半精度。
        """
        # 参数含义：
        # - network：要转换成 FP16 的模型对象。
        #
        # 目标：
        # - 大部分层转成 FP16，提高速度、节省显存；
        # - BatchNorm 层保持 FP32，减少数值不稳定。

        # nn.Sequential(...) 会把多个模块按顺序串起来。
        #
        # 这里串了两部分：
        # 1. self.tofp16()
        #    先把输入张量转成 FP16；
        # 2. self.BN_convert_float(network.half())
        #    network.half() 先把整个模型转成 FP16，
        #    BN_convert_float(...) 再把其中的 BatchNorm 层转回 FP32。
        #
        # 所以最终得到的是一个“输入先转 FP16，再进入模型”的新模型包装。
        return nn.Sequential(self.tofp16(),
                             self.BN_convert_float(network.half()))

    def warmup_learning_rate(self, init_lr, no_of_steps, epoch, len_epoch):
        """前 5 个 epoch 使用学习率预热。"""
        # 参数含义：
        # - init_lr：初始学习率，也就是 main.py 里 --lr 传进来的值
        # - no_of_steps：总 epoch 数，也就是 main.py 里的 --steps
        # - epoch：当前 epoch 编号，从 0 开始
        # - len_epoch：一个 epoch 里有多少个 batch，也就是 len(trainloader)
        #
        # 这个函数返回当前 warmup 阶段应该使用的学习率。
        #
        # 学习率预热的直觉：
        # 训练刚开始时，模型参数是随机的。
        # 如果一上来就用很大的学习率，参数可能被更新得太猛。
        # warmup 会让学习率从较小值逐渐升高。

        factor = no_of_steps // 30
        # // 是整数除法。
        # no_of_steps // 30 表示总训练 epoch 数除以 30 后向下取整。
        #
        # 例如 no_of_steps=200 时：
        # factor = 200 // 30 = 6。
        #
        # 这个 factor 会影响下面初始 lr 的缩放。

        lr = init_lr * (0.1**factor)
        # ** 表示乘方。
        # 0.1**factor 表示 0.1 的 factor 次方。
        #
        # 这里先把 init_lr 缩小很多，作为 warmup 起点附近的学习率。

        # 执行预热计算。
        lr = lr * float(1 + epoch + no_of_steps * len_epoch) / (5. * len_epoch)
        # 这一行根据当前 epoch、总 epoch 数、每个 epoch 的 batch 数计算 warmup 学习率。
        #
        # float(...) 把结果转成浮点数。
        # 5. 等价于 5.0，表示前 5 个 epoch 做预热。
        #
        # 简单理解：
        # 这个公式会让 lr 在训练开始的前 5 个 epoch 内逐步变化，
        # 而不是一开始就直接使用 self.lr。

        # 返回计算出的学习率，调用处会把它写入 optimizer.param_groups。
        return lr

    def train(self, epoch, no_of_steps, trainloader):
        # train(...) 是 Trainer 类里的一个方法，用来训练一个 epoch。
        # 参数含义：
        # - self：当前 Trainer 对象本身
        # - epoch：当前是第几个 epoch，注意这里从 0 开始计数
        # - no_of_steps：总共要训练多少个 epoch
        # - trainloader：训练集的数据加载器，会一批一批提供训练图片和标签
        # 
        # 训练集和测试集最大的区别就在这里：
        # - train() 会执行 loss.backward() 和 optimizer.step() 更新模型参数
        # - evaluate() 不会更新参数，只看模型表现

        ############################ 设置模型为训练模式 ################################
        self.model.train()
        # self.model.train() 是 PyTorch 模型对象的方法。
        # 它不是“开始训练整个流程”的意思，而是把模型切换到训练模式。
        #
        # 有些层在训练和测试时行为不同，例如：
        # - BatchNorm：训练时会更新 batch 的统计量
        # - Dropout：训练时会随机丢弃一部分神经元
        #
        # 所以训练前要调用 self.model.train()。

        train_loss, correct, total = 0, 0, 0
        # 同时给三个变量赋初值。
        # (语法) a, b, c = 0, 0, 0 表示：
        # - train_loss = 0：累计训练 loss
        # - correct = 0：累计预测正确的样本数
        # - total = 0：累计已经处理过的样本数

        ############################ 设置当前 epoch 的学习率 ############################
        # 如果 epoch 小于 5，使用学习率预热；否则使用学习率调度器。
        if epoch < 5:
            # 训练刚开始时，模型参数还是随机初始化的。
            # 如果一开始学习率太大，参数可能被更新得太猛，训练不稳定。
            # 所以前 5 个 epoch 先使用较小学习率，再逐渐升上去。

            lr = self.warmup_learning_rate(self.lr, no_of_steps, epoch, len(trainloader))
            # warmup_learning_rate(...) 返回当前 epoch 应该使用的学习率。
            # len(trainloader) 表示训练集中一共有多少个 batch。

            for param_group in self.optimizer.param_groups:
                # optimizer.param_groups 是优化器管理的一组参数配置。
                # 简单理解：这里是在遍历优化器里所有需要设置学习率的参数组。
                #
                # PyTorch 的优化器可以同时管理多组参数。
                # 每一组参数都可以有自己的学习率、动量、权重衰减等配置。
                #
                # 当前代码创建优化器时写的是：
                #   optim.SGD(self.model.parameters(), self.lr, momentum=0.9, weight_decay=5e-4)
                #   或者 FP16 模式下：
                #   optim.SGD(self.master_params, self.lr, momentum=0.9, weight_decay=5e-4)
                # 这种写法通常只有一组参数，所以这个 for 循环通常只执行一次。
                #
                # 但 PyTorch 也允许这样分组：
                # optimizer = optim.SGD([
                #     {'params': model.layer1.parameters(), 'lr': 0.01},
                #     {'params': model.layer2.parameters(), 'lr': 0.001},
                # ])
                # 这样 layer1 和 layer2 就可以用不同学习率训练。
                #
                # 所以这里写 for 循环，是为了兼容“优化器里可能有多组参数”的情况。
                # 当前只有一组时，它就只改这一组的 lr；
                # 以后如果有多组，它会把每一组的 lr 都改掉。

                param_group['lr'] = lr # 把当前参数组的学习率改成 warmup 算出来的 lr。
        elif epoch == 5:
            # 第 5 个 epoch 时，把学习率恢复为初始设定的 self.lr。
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        # 另一种学习率调度写法，当前代码没有启用。
        # scheduler = MultiStepLR(self.optimizer, milestones=[80, 120, 160, 180], gamma=0.1)
        # if epoch >= 5:
        #     scheduler.step(epoch=epoch)
        #
        # 注意：这里说“当前代码没有启用”，指的是上面这三行被注释掉的备用写法没有启用。
        # 不是说 Trainer.__init__() 里创建的 self.scheduler 没有用。
        #
        # 当前真正生效的学习率逻辑是：
        # 1. 在 Trainer.__init__() 里创建 self.scheduler：
        #    MultiStepLR(self.optimizer, milestones=[10, 20, 50, 100, 180], gamma=0.1)
        # 2. 在 train() 开头，如果 epoch < 5：
        #    使用 warmup_learning_rate(...) 手动设置较小的学习率。
        #    这叫学习率预热，目的是让训练刚开始时不要更新得太猛。
        # 3. 如果 epoch == 5：手动把学习率恢复成初始学习率 self.lr
        # 4. 在这个 train() 函数的最后：
        #    if epoch >= 5:
        #        self.scheduler.step()
        #    这里才是真正调用 __init__() 里那个 self.scheduler。
        #
        # 所以整体关系是：
        # - 前 5 个 epoch：不用 MultiStepLR，先手动 warmup；
        # - 第 5 个 epoch：恢复到初始学习率；
        # - 第 5 个 epoch 之后：每个 epoch 末尾调用 self.scheduler.step()；
        #   - 当 scheduler 走到 milestones=[10, 20, 50, 100, 180] 时，学习率会乘以 gamma=0.1。
        #
        # 这里还有一个容易混淆的点：
        # 被注释掉的备用写法 milestones 是 [80, 120, 160, 180]；
        # 当前实际使用的 self.scheduler milestones 是 [10, 20, 50, 100, 180]。

        print('Learning Rate: %g' % (list(map(lambda group: group['lr'], self.optimizer.param_groups)))[0])
        # 打印当前学习率。
        #
        # (语法)
        # map(lambda group: group['lr'], self.optimizer.param_groups)：
        # - map(...) 会把后面每个参数 group 传给前面的 lambda 函数；
        # - lambda group: group['lr'] 是一个匿名函数，
        #   输入是参数组 group
        #   输出是这个参数组的学习率 group['lr']
        # - 结果是一个 map 对象，里面保存了每个参数组的学习率。
        #
        # (语法)
        # list(...) 把 map 对象转换成列表。
        # - [0] 取出列表里的第一个元素，也就是当前训练使用的学习率。

        ############################ 定义损失函数 ######################################
        # 损失函数使用 FP32 计算。
        criterion = nn.CrossEntropyLoss()
        # criterion 是变量，保存一个交叉熵损失函数对象。
        #
        # CrossEntropyLoss 用于多分类任务。
        # 在这个实验里，模型输出 10 个类别的 logits，
        # targets 代表真实类别编号，例如 0~9。

        ############################ 遍历训练集的每一个 batch ##########################
        for idx, (inputs, targets) in enumerate(trainloader):
            # for 循环会从 trainloader 里一批一批取数据。
            #
            # (语法) enumerate(trainloader)：
            # - idx 是当前 batch 的编号，从 0 开始；
            # - inputs 是这一批图片；
            # - targets 是这一批图片对应的真实标签。
            #
            # 对 CIFAR-10 来说：
            # - inputs 的形状大致是 [batch_size, 3, 32, 32]
            # - targets 的形状大致是 [batch_size]

            if self.train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
                # 如果使用 GPU，就把这一批图片和标签都移动到 GPU。
                # 模型在 GPU 上时，输入数据也必须在 GPU 上。

            self.model.zero_grad()
            # 清空上一轮 batch 留下的梯度。
            #
            # PyTorch 默认会累加梯度。
            # 如果不清零，当前 batch 的梯度会和上一个 batch 的梯度混在一起。

            outputs = self.model(inputs)
            # 前向传播 forward: 把图片 inputs 输入模型，得到 outputs。
            # - outputs 不是最终的类别名字，而是每个类别的分数 logits。
            #   e.g. 一张图片可能输出 10 个数，分别对应 10 个 CIFAR-10 类别。

            # 损失值使用 FP32 计算，因为归约类操作用 FP16 表示时可能不准确。
            loss = criterion(outputs, targets)
            # 用模型预测 outputs 和真实标签 targets 计算 loss。
            # 注意：这里是“训练集预测结果 vs 训练集真实标签”，不是和测试集比较。
            if self.loss_scaling:
                # 有时 loss 会小到难以用 FP16 表示，
                # 所以这里用一个较大的 2 的幂对 loss 做缩放，当前使用 2**7。
                loss = loss * self._LOSS_SCALE
                # 缩放的是 loss 这个损失值。
                # 这样 loss.backward() 得到的梯度也会被临时放大，
                # 减少 FP16 反向传播时梯度太小变成 0 的风险。

            # 反向传播 backward，根据 loss 计算模型每个参数的梯度。
            loss.backward()
            # 梯度可以理解成：
            # 为了让 loss 变小，每个参数应该往哪个方向、改多少。

            if self.fp16_mode: # FP16 模式下，模型参数负责计算，FP32 主参数负责更新.

                # 把刚算出的梯度移动到 FP32 主参数上，这样后续可以用 FP32 执行梯度更新。
                self.model_grads_to_master_grads(self.model_params, self.master_params)
                # 这一步之后，FP32 的 self.master_params 上有了梯度。

                if self.loss_scaling:
                    # 如果前面放大过 loss，现在需要把梯度按相同比例缩回去，
                    # 因为此时梯度已经在 FP32 主参数上。
                    for params in self.master_params:
                        params.grad.data = params.grad.data / self._LOSS_SCALE

                # 用 FP32 执行权重更新。
                self.optimizer.step()
                # optimizer.step() 是真正修改参数的动作。
                # 在 FP16 模式下，它修改的是 FP32 master_params。

                # 把更新后的权重复制回 FP16 模型权重。
                self.master_params_to_model_params(self.model_params, self.master_params)
                # 这样下一次 forward 时，FP16 模型用的就是更新后的参数。
            else:
                self.optimizer.step()
                # 普通 FP32 模式下，优化器直接更新 self.model.parameters()。

            ############################ 统计训练 loss 和准确率 ########################
            train_loss += loss.item()
            # loss.item() 把只有一个数的 Tensor 转成 Python 数字。
            # train_loss 用来累计当前 epoch 到目前为止的 loss。

            _, predicted = outputs.max(1)
            # outputs.max(1) 会在“类别维度”上找最大值。
            # 返回：最大分数, 最大分数对应的类别编号
            #
            # 这里用 _ 接住最大分数，表示后面不用它；
            # predicted 保存预测出来的类别编号。

            total += targets.size(0)
            # targets.size(0) 是当前 batch 的样本数量。
            # total 累加已经看过多少张训练图片。

            correct += (targets == predicted).sum().item()
            # targets == predicted 会得到一组 True/False：
            # True 表示这张图预测对了, False 表示预测错了
            #
            # sum() 会把 True 当作 1、False 当作 0 来求和，
            # 得到当前 batch 预测正确的数量。

            progress_bar( idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (train_loss / (idx + 1), 100. * correct / total, correct, total))
            # 打印训练进度条。
            # 显示当前平均 loss、训练准确率、正确数量/总数量。

        if epoch >= 5: # 2020.09.09 修改：第 5 个 epoch 之后再更新调度器。
            self.scheduler.step()
            # 更新学习率调度器：到 milestones 指定的 epoch 时，学习率会乘以 gamma。

    def evaluate(self, epoch, testloader):
        # evaluate(...) 是 Trainer 类里的一个方法，用来在测试集上评估模型。
        # 参数含义：
        # - epoch：当前是第几个 epoch
        # - testloader：测试集的数据加载器，会一批一批提供测试图片和标签
        #
        # 这个函数不会更新模型参数。
        # 它的作用是“考试”：看当前模型在没参与训练的数据上表现如何。

        ############################ 设置模型为评估模式 ################################
        self.model.eval()
        # 把模型切换到评估模式。
        #
        # 和 train() 相反：
        # - BatchNorm 不再用当前 batch 更新统计量
        # - Dropout 不再随机丢弃神经元
        # 这样测试结果更稳定。

        # 初始化测试阶段的统计变量：
        test_loss = 0 # 累计测试 loss
        correct = 0 # 累计预测正确的测试样本数
        total = 0 # 累计测试样本总数

        criterion = nn.CrossEntropyLoss()
        # 测试阶段也用交叉熵损失。
        # 注意：这里的 loss 是“测试集预测结果 vs 测试集真实标签”。
        # 它不参与参数更新，只用于观察模型表现。

        with torch.no_grad():
            # torch.no_grad() 表示下面这段代码不记录梯度。
            #
            # 测试时不需要反向传播，也不需要 optimizer.step()。
            # 关闭梯度记录可以节省显存和计算量。

            for idx, (test_x, test_y) in enumerate(testloader):
                # 从测试集 testloader 中一批一批取数据。
                # test_x 是测试图片，test_y 是测试标签。

                if self.train_on_gpu: # 如果模型在 GPU 上，测试数据也要移动到 GPU。
                    test_x, test_y = test_x.cuda(), test_y.cuda()
                    
                outputs = self.model(test_x)
                # 前向传播：用当前模型对测试图片做预测。
                # 这里只预测，不训练。

                loss = criterion(outputs, test_y)
                # 计算测试 loss。
                # 注意：不会调用 loss.backward()，所以不会根据测试集修改模型。

                test_loss += loss.item()
                # 累计测试 loss。

                _, predicted = outputs.max(1)
                # 取每张测试图片分数最高的类别，作为预测类别。

                total += test_y.size(0)
                # 累计测试样本数量

                correct += (predicted == test_y).sum().item()
                # 累计预测正确的测试样本数量

                progress_bar(
                    idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (loss / (idx + 1), 100. * correct / total, correct, total))
                # 打印测试进度条。
                # 这里显示的是测试集上的 loss 和准确率。

        acc = 100.0 * correct / total # 计算当前 epoch 的测试准确率，单位是百分比。
        if acc > self.best_acc:
            self.save_model(self.model, self.model_name, acc, epoch)
            # 如果当前测试准确率超过历史最好准确率，就保存模型。
            # 所以 weights/ 里保存的是测试表现更好的模型参数。

    def save_model(self, model, model_name, acc, epoch):
        # save_model(...) 用来保存当前模型权重。
        #
        # 参数含义：
        # - model：要保存的模型对象
        # - model_name：模型名字，例如 'resnet18'
        # - acc：当前模型在测试集上的准确率
        # - epoch：当前 epoch 编号
        #
        # 这个函数在 evaluate() 里被调用：
        # 如果当前测试准确率 acc 超过历史最好 self.best_acc，
        # 就保存一次模型。

        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # state 是一个字典 dict，用来打包要保存的信息。
        #
        # (语法) 字典用 {key: value} 表示。
        #
        # 这里保存了三项：
        # - 'net'：模型参数
        # - 'acc'：保存时的测试准确率
        # - 'epoch'：保存时是第几个 epoch
        #
        # model.state_dict() 是 PyTorch 模型对象的方法。
        # 它返回模型所有可学习参数和 buffer 的字典。
        # 真正恢复模型时，主要靠这里的 'net'。

        if self.fp16_mode:
            # 如果当前是 FP16 模式，权重保存到 weights/<model_name>_fp16/ 目录。

            save_name = os.path.join('weights', model_name + '_fp16',
                                     'weights.%03d.%.03f.pt' % (epoch, acc))
        else:
            # 普通 FP32 模式，权重保存到 weights/<model_name>/ 目录。

            save_name = os.path.join('weights', model_name,
                                     'weights.%03d.%.03f.pt' % (epoch, acc))
        # os.path.join(...) 用来拼接路径。
        # 这样比手写 'weights/' + model_name 更稳妥。
        #
        # 'weights.%03d.%.03f.pt' % (epoch, acc) 是字符串格式化：
        # - %03d：把 epoch 格式化成至少 3 位整数，不足补 0，例如 7 -> 007
        # - %.03f：把 acc 格式化成保留 3 位小数
        #
        # 例如可能得到：
        # weights/resnet18/weights.012.84.530.pt

        if not os.path.exists(os.path.dirname(save_name)):
            # os.path.dirname(save_name) 取出文件所在目录。
            # 例如 save_name 是 weights/resnet18/weights.012.84.530.pt，
            # dirname 就是 weights/resnet18。
            #
            # os.path.exists(...) 检查这个目录是否已经存在。
            # not 表示取反。

            os.makedirs(os.path.dirname(save_name))
            # 如果目录不存在，就递归创建目录。
            # makedirs 可以一次创建多级目录，例如 weights/resnet18。

        torch.save(state, save_name)
        # torch.save(...) 把 state 保存到磁盘文件。
        # 保存出来的是 .pt 文件，后面可以用 torch.load(...) 读回来。

        print("\nSaved state at %.03f%% accuracy. Prev accuracy: %.03f%%" %
              (acc, self.best_acc))
        # 打印保存提示：
        # 当前准确率是多少，之前最好准确率是多少。

        self.best_acc = acc
        # 更新历史最好准确率。

        self.best_epoch = epoch
        # 更新历史最好准确率对应的 epoch。

    def load_model(self, path=None):
        """
        加载之前保存的模型。这里不会检查精度类型。
        """
        # load_model(...) 用来从磁盘加载之前保存的权重。
        #
        # 参数含义：
        # - path：可选参数。
        #   如果传了 path，就加载指定路径的模型文件；
        #   如果没有传，就尝试根据 self.best_epoch / self.best_acc 拼出默认路径。
        #
        # 注意：
        # 这个函数当前代码里没有在 main.py 中直接调用。
        # 它是一个工具函数，方便以后恢复训练或测试已有权重。

        if path is not None:
            # 如果用户显式传入了路径，就直接使用这个路径。

            checkpoint_name = path
        elif self.fp16_mode:
            # 如果没有传 path，并且当前是 FP16 模式，
            # 就尝试去 weights/<model_name>_fp16/ 目录里找最好模型。

            checkpoint_name = os.path.join(
                'weights', self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        else:
            # 普通 FP32 模式下，理论上应该去 weights/<model_name>/ 目录找。
            #
            # 但这里原代码写的是 self.model_name + '_fp16'，
            # 也就是说它仍然会去 *_fp16 目录找。
            # 这很可能是原代码里的一个小 bug。
            # 因为当前主训练流程没有调用 load_model()，所以不影响 main.py 训练。

            checkpoint_name = os.path.join(
                'weights', self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        if not os.path.exists(checkpoint_name):
            # 如果目标权重文件不存在，就打印提示并返回。

            print("Best model not found")
            # return 表示提前结束函数。
            return

        checkpoint = torch.load(checkpoint_name)
        # torch.load(...) 从 .pt 文件里读取保存的数据。
        # 这里读出来的 checkpoint 是前面 save_model() 保存的 state 字典。

        self.model.load_state_dict(checkpoint['net'])
        # load_state_dict(...) 把 checkpoint['net'] 里的参数加载进当前模型。
        #
        # checkpoint['net'] 对应 save_model() 里的：
        # 'net': model.state_dict()

        self.best_acc = checkpoint['acc']
        # 恢复保存时的最好准确率。

        self.best_epoch = checkpoint['epoch']
        # 恢复保存时的 epoch。

        print("Loaded Model with accuracy: %.3f%%, from epoch: %d" %
              (checkpoint['acc'], checkpoint['epoch'] + 1))
        # 打印加载结果。
        # checkpoint['epoch'] 是从 0 开始计数，所以显示给人看时加 1。

    def train_and_evaluate(self, traindataloader, testdataloader, no_of_steps):
        # train_and_evaluate(...) 是整个训练流程的外层循环。
        #
        # 参数含义：
        # - traindataloader：训练集 DataLoader
        # - testdataloader：测试集 DataLoader
        # - no_of_steps：训练多少个 epoch
        #
        # 这个函数在 main.py 的最后被调用：
        # trainer.train_and_evaluate(trainloader, testloader, args.steps)

        self.best_acc = 0.0
        # 开始训练前，把历史最好准确率重置为 0。
        # 后面 evaluate() 如果发现更高准确率，会保存模型并更新 self.best_acc。

        for i in range(no_of_steps):
            # (语法) range(no_of_steps) 会生成 0 到 no_of_steps-1 的整数序列。
            #
            # 如果 no_of_steps=200，那么 i 会依次是：
            # 0, 1, 2, ..., 199
            #
            # 每一次循环就是一个 epoch。

            print('\nEpoch: %d' % (i + 1))
            # 打印当前 epoch。
            # i 从 0 开始，但显示给人看通常从 1 开始，所以用 i + 1。

            self.train(i, no_of_steps, traindataloader)
            # 先在训练集上训练一个 epoch。
            # 这里会更新模型参数。

            self.evaluate(i, testdataloader)
            # 再在测试集上评估一次。
            # 这里不会更新模型参数。
            #
            # 如果测试准确率比历史最好高，evaluate() 里会调用 save_model() 保存权重。
