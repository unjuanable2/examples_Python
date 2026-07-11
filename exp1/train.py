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

        第一套：model_params
        - 来自 model.parameters()
        - 是模型里真正参与前向传播和反向传播的参数
        - 在 FP16 模式下，它们大部分是半精度参数

        第二套：master_params
        - 是从 model_params 复制出来的一份 FP32 参数
        - 不直接参与模型 forward
        - 专门交给 optimizer 更新

        为什么不只保留 FP16 参数？
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
        # master_params 也是一个列表。
        # 它是从 model_params 复制出来的一份 FP32 参数。
        #
        # p.detach()：从原计算图里分离出来，避免和原参数共享梯度历史；
        # clone()：复制一份新的数据；
        # float()：转成 FP32。

        for p in master_params:
            p.requires_grad = True
            # master_params 后面要被优化器更新，所以也需要梯度。

        return model_params, master_params
        # 返回两套参数。
        # 调用处用：
        # self.model_params, self.master_params = self.prep_param_list(self.model)
        # 分别接住这两个返回值。

    def master_params_to_model_params(self, model_params, master_params):
        """
        把 FP32 主参数复制回 FP16 模型参数。
        """
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

    def model_grads_to_master_grads(self, model_params, master_params):
        for model, master in zip(model_params, master_params):
            if master.grad is None:
                master.grad = Variable(master.data.new(*master.data.size()))
            master.grad.data.copy_(model.grad.data)

    def BN_convert_float(self, module):
        '''
        配合 network_to_half 使用。
        BatchNorm 层需要保持单精度参数。
        这里递归查找所有子层，并把 BatchNorm 层转回 float。
        不能直接使用内置的 .apply，因为 .apply 会把函数应用到所有模块、参数和缓冲区，
        这样就无法根据模块类型只保护 BatchNorm 的 float 转换。
        '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.float()
        for child in module.children():
            self.BN_convert_float(child)
        return module

    class tofp16(nn.Module):
        """
        添加一个层，用来把输入转换成 FP16。
        这是一个模型包装层，实现的逻辑相当于：
            def forward(self, input):
                return input.half()
        """

        def __init__(self):
            super(Trainer.tofp16, self).__init__()

        def forward(self, input):
            return input.half()

    def network_to_half(self, network):
        """
        用对 BatchNorm 安全的方式把模型转换成半精度。
        """
        return nn.Sequential(self.tofp16(),
                             self.BN_convert_float(network.half()))

    def warmup_learning_rate(self, init_lr, no_of_steps, epoch, len_epoch):
        """前 5 个 epoch 使用学习率预热。"""
        factor = no_of_steps // 30
        lr = init_lr * (0.1**factor)
        """执行预热计算。"""
        lr = lr * float(1 + epoch + no_of_steps * len_epoch) / (5. * len_epoch)
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

                param_group['lr'] = lr
                # 把当前参数组的学习率改成 warmup 算出来的 lr。
        elif epoch == 5:
            # 第 5 个 epoch 时，把学习率恢复为初始设定的 self.lr。
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        # 另一种学习率调度写法，当前代码没有启用。
        # scheduler = MultiStepLR(
        #     self.optimizer, milestones=[80, 120, 160, 180], gamma=0.1)
        # if epoch >= 5:
        #     scheduler.step(epoch=epoch)

        print('Learning Rate: %g' % (list(
            map(lambda group: group['lr'], self.optimizer.param_groups)))[0])
        # 打印当前学习率。
        #
        # 这里语法比较绕，可以先理解为：
        # - 从 optimizer.param_groups 里取出当前学习率
        # - 用 print 打印出来

        ############################ 定义损失函数 ######################################
        # 损失函数使用 FP32 计算。
        criterion = nn.CrossEntropyLoss()
        # criterion 是变量，保存一个交叉熵损失函数对象。
        #
        # CrossEntropyLoss 用于多分类任务。
        # 在这个实验里，模型输出 10 个类别的 logits，
        # targets 是真实类别编号，例如 0~9。
        #
        # loss 的含义：
        # - 如果模型给真实类别的分数高，loss 小；
        # - 如果模型给错误类别的分数高，loss 大。

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
            # 前向传播 forward。
            # 把图片 inputs 输入模型，得到 outputs。
            #
            # outputs 不是最终的类别名字，而是每个类别的分数 logits。
            # 例如一张图片可能输出 10 个数，分别对应 10 个 CIFAR-10 类别。

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

            # 计算梯度。
            loss.backward()
            # 反向传播 backward。
            # 根据 loss 计算模型每个参数的梯度。
            #
            # 梯度可以理解成：
            # 为了让 loss 变小，每个参数应该往哪个方向、改多少。

            if self.fp16_mode:
                # FP16 模式下，模型参数负责计算，FP32 主参数负责更新。

                # 把刚算出的梯度移动到 FP32 主参数上，
                # 这样后续可以用 FP32 执行梯度更新。
                self.model_grads_to_master_grads(self.model_params,
                                                 self.master_params)
                # 这一步之后，FP32 的 self.master_params 上有了梯度。

                if self.loss_scaling:
                    # 如果前面放大过 loss，现在需要把梯度按相同比例缩回去，
                    # 因为此时梯度已经在 FP32 主参数上。
                    for params in self.master_params:
                        params.grad.data = params.grad.data / self._LOSS_SCALE
                        # 前面 loss 乘了 128，所以梯度也被放大了 128。
                        # 真正更新参数前，要把梯度除以 128，恢复真实大小。

                # 用 FP32 执行权重更新。
                self.optimizer.step()
                # optimizer.step() 是真正修改参数的动作。
                # 在 FP16 模式下，它修改的是 FP32 master_params。

                # 把更新后的权重复制回 FP16 模型权重。
                self.master_params_to_model_params(self.model_params,
                                                   self.master_params)
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
            # 返回两个东西：
            # - 最大分数
            # - 最大分数对应的类别编号
            #
            # 这里用 _ 接住最大分数，表示后面不用它；
            # predicted 保存预测出来的类别编号。

            total += targets.size(0)
            # targets.size(0) 是当前 batch 的样本数量。
            # total 累加已经看过多少张训练图片。

            correct += (targets == predicted).sum().item()
            # targets == predicted 会得到一组 True/False：
            # - True 表示这张图预测对了
            # - False 表示预测错了
            #
            # sum() 会把 True 当作 1、False 当作 0 来求和，
            # 得到当前 batch 预测正确的数量。

            progress_bar(
                idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (train_loss / (idx + 1), 100. * correct / total, correct,
                 total))
            # 打印训练进度条。
            # 显示当前平均 loss、训练准确率、正确数量/总数量。

        if epoch >= 5: # 2020.09.09 修改：第 5 个 epoch 之后再更新调度器。
            self.scheduler.step()
            # 更新学习率调度器。
            # 到 milestones 指定的 epoch 时，学习率会乘以 gamma。

    def evaluate(self, epoch, testloader):
        # evaluate(...) 是 Trainer 类里的一个方法，用来在测试集上评估模型。
        #
        # 参数含义：
        # - epoch：当前是第几个 epoch
        # - testloader：测试集的数据加载器，会一批一批提供测试图片和标签
        #
        # 这个函数不会更新模型参数。
        # 它的作用是“考试”：看当前模型在没参与训练的数据上表现如何。

        ############################ 设置模型为评估模式 ################################
        self.model.eval()
        # self.model.eval() 把模型切换到评估模式。
        #
        # 和 train() 相反：
        # - BatchNorm 不再用当前 batch 更新统计量
        # - Dropout 不再随机丢弃神经元
        #
        # 这样测试结果更稳定。

        test_loss = 0
        correct = 0
        total = 0
        # 初始化测试阶段的统计变量：
        # - test_loss：累计测试 loss
        # - correct：累计预测正确的测试样本数
        # - total：累计测试样本总数

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
                # 从测试集 DataLoader 中一批一批取数据。
                # test_x 是测试图片，test_y 是测试标签。

                if self.train_on_gpu:
                    test_x, test_y = test_x.cuda(), test_y.cuda()
                    # 如果模型在 GPU 上，测试数据也要移动到 GPU。

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
                # 累计测试样本数量。

                correct += (predicted == test_y).sum().item()
                # 累计预测正确的测试样本数量。

                progress_bar(
                    idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (loss / (idx + 1), 100. * correct / total, correct, total))
                # 打印测试进度条。
                # 这里显示的是测试集上的 loss 和准确率。

        acc = 100.0 * correct / total
        # 计算当前 epoch 的测试准确率，单位是百分比。

        if acc > self.best_acc:
            self.save_model(self.model, self.model_name, acc, epoch)
            # 如果当前测试准确率超过历史最好准确率，就保存模型。
            # 所以 weights/ 里保存的是测试表现更好的模型参数。

    def save_model(self, model, model_name, acc, epoch):
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if self.fp16_mode:
            save_name = os.path.join('weights', model_name + '_fp16',
                                     'weights.%03d.%.03f.pt' % (epoch, acc))
        else:
            save_name = os.path.join('weights', model_name,
                                     'weights.%03d.%.03f.pt' % (epoch, acc))

        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))

        torch.save(state, save_name)
        print("\nSaved state at %.03f%% accuracy. Prev accuracy: %.03f%%" %
              (acc, self.best_acc))
        self.best_acc = acc
        self.best_epoch = epoch

    def load_model(self, path=None):
        """
        加载之前保存的模型。这里不会检查精度类型。
        """
        if path is not None:
            checkpoint_name = path
        elif self.fp16_mode:
            checkpoint_name = os.path.join(
                'weights', self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        else:
            checkpoint_name = os.path.join(
                'weights', self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        if not os.path.exists(checkpoint_name):
            print("Best model not found")
            return
        checkpoint = torch.load(checkpoint_name)
        self.model.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.best_epoch = checkpoint['epoch']
        print("Loaded Model with accuracy: %.3f%%, from epoch: %d" %
              (checkpoint['acc'], checkpoint['epoch'] + 1))

    def train_and_evaluate(self, traindataloader, testdataloader, no_of_steps):
        self.best_acc = 0.0
        for i in range(no_of_steps):
            print('\nEpoch: %d' % (i + 1))
            self.train(i, no_of_steps, traindataloader)
            self.evaluate(i, testdataloader)
