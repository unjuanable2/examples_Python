"""根据 BN gamma 对 YOLOv3 做结构化通道剪枝。

老师已经提供 ``sparse-yolov3-full-mAP48.1.pt``，所以本作业从稀疏模型开始：

1. 收集可剪枝 BatchNorm 层的 gamma（即 BN 的可学习缩放参数 weight）；
2. 将全部 gamma 的绝对值排序，在 ``--percent 0.5`` 位置确定全局阈值；
3. 小于阈值的通道 mask 设为 0，并在 COCO 5k 上观察直接剪枝后的 mAP；
4. 修改 cfg 中各卷积层 filters，创建物理通道数真正减少的 compact_model；
5. 把保留通道的卷积/BN 参数复制进 compact_model，再次验证并保存 cfg/weights。

脚本不会调用 train.py，因此没有剪枝后微调，符合本次作业要求。
"""

from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse



if __name__ == '__main__':  # python shortcut_prune.py --percent 0.5
    # argparse 将 run_exp.sh 传入的文本参数转换为 Python 对象 opt。
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/sparse-yolov3-full-mAP48.1.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.5, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='COCO validation batch size; reduce it when GPU memory is insufficient')
    parser.add_argument('--device', default='0', help='CUDA device id, e.g. 0; cpu is supported but slow')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    # 复用 detect.py/test.py 相同的设备选择逻辑，避免机器有 GPU 时无法选择编号。
    device = torch_utils.select_device(opt.device)
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ',opt.weights)


    #****************************************************************
    # add the prune code here
    # test() 接收内存中的 model，因此不会再次从磁盘加载权重。
    # 返回值第一个元素依次含 P、R、mAP、F1 等指标。
    eval_model = lambda model:test(model=model,cfg=opt.cfg, data=opt.data,
                                   batch_size=opt.batch_size, img_size=img_size)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    origin_nparameters = obtain_num_parameters(model)
    
    CBL_idx, Conv_idx, prune_idx,shortcut_idx,shortcut_all= parse_module_defs2(model.module_defs)

    # gather_bn_weights() 把所有可剪枝 BN 层的 |gamma| 拼成一个一维 tensor。
    # gamma 越接近 0，该通道对 BN 输出的缩放越弱，因此被视为越不重要。
    bn_weights = gather_bn_weights(model.module_list, prune_idx) #get gamma

    sorted_bn = torch.sort(bn_weights)[0] #sort gamma torch.max torch.softmax

    def prune_and_eval(model, sorted_bn, percent=.0):
        model_copy = deepcopy(model) 
        # percent=0.5 表示取全局排序中第 50% 个 gamma 作为阈值，不代表最终参数量
        # 或权重文件大小一定恰好下降 50%。shortcut 约束也会影响实际保留通道数。
        thre_index = int(len(sorted_bn)*percent)  # 排序后取指定比例位置的下标
        thre1 = sorted_bn[thre_index]  # 取得对应的全局 gamma 阈值
        print(f'Channels with Gamma value less than {thre1:.10f} are pruned!')

        total, remain_num = 0, 0
        idx_new = dict()
        for idx in prune_idx:
            if idx not in shortcut_idx:
                bn_module  = model_copy.module_list[idx][1] #獲取可剪zhi的卷基層對應的BN層
                mask = obtain_bn_mask(bn_module, thre1)
                idx_new[idx] = mask
                bn_module.weight.data.mul_(mask) #a,b,c,d  0,1,1,0 = 0,b,c,0
            else:
                bn_module  = model_copy.module_list[idx][1]
                mask = idx_new[shortcut_idx[idx]]
                idx_new[idx]=mask
                bn_module.weight.data.mul_(mask)
            remain_num += int(mask.sum()) # 0 1 0 1 0  sum=2
            total+=mask.shape[0] # 0 1 0 1 0, shape[0] = 5
        print(f'Num. of channels has been reduced from {total} to {remain_num}')
        
        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]
        return thre1
    percent = opt.percent
    threshold = prune_and_eval(model, sorted_bn, percent)
    #****************************************************************
    #虽然上面已经能看到剪枝后的效果，但是没有生成剪枝后的模型结构，因此下面的代码是为了生成新的模型结构并拷贝旧模型参数到新模型

    def obtain_filters_mask(model, thre, CBL_idx, prune_idx):
        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        idx_new=dict()
        #CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:
                if idx not in shortcut_idx:

                    mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
                    idx_new[idx]=mask
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                else:
                    mask=idx_new[shortcut_idx[idx]]
                    idx_new[idx]=mask
                    remain= int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain                
                if remain == 0:
                    print("Channels would be all pruned!")
                    # raise Exception
                    max_value = bn_module.weight.data.abs().max()
                    mask = obtain_bn_mask(bn_module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                        f'remaining channel: {remain:>4d}')
            else:
                mask = np.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]
            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())
        #因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数
        prune_ratio = pruned / total
        print(f'Prune channels: {pruned}/{total}\tPrune ratio: {prune_ratio:.3f}')
        return num_filters, filters_mask

    num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)
    #CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
    #获得原始模型的module_defs，并修改该defs中的卷积核数量
    compact_module_defs = deepcopy(model.module_defs)
    for idx, num in zip(CBL_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)

    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

    # 随机输入只用于测网络前向耗时和确认两个内部模型可运行，不用于计算 mAP。
    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):
        model.eval()
        # CUDA 操作异步执行；计时前后同步，才能测到真实完成时间。
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        avg_infer_time = (time.time() - start) / repeat
        return avg_infer_time, output

    print('testing Inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing final model')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)

    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        #["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["mAP", f'{0.481}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{percent}_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.weights.replace('/', f'/prune_{percent}_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')

