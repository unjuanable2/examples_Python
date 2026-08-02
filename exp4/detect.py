"""使用 YOLOv3 对图片或视频做目标检测。 

3. load_darknet_weights() 把 .weights 参数加载到对应网络；
4. LoadImages 逐帧读取图片/视频并执行 letterbox、BGR→RGB 和 0~1 归一化；
5. model(img) 前向传播，non_max_suppression() 删除重复检测框；
6. scale_coords() 将 416×416 输入坐标映射回原视频，plot_one_box() 画框；
7. OpenCV VideoWriter 保存带检测框、模型名称和当前 FPS 的结果视频。

原模型和剪枝模型共用本文件，区别只来自 --cfg 与 --weights 的正确配对。
"""

import argparse
# argparse 是 Python 标准库，用于定义和读取 ``--cfg``、``--weights`` 等命令行参数。

from sys import platform
# platform 是当前操作系统名称；这里只用它判断是否为 macOS（darwin）。

from models import *  # set ONNX_EXPORT in models.py
# models.py 提供 Darknet、权重加载函数和 ONNX_EXPORT 开关。

from utils.datasets import *
# utils/datasets.py 提供 LoadImages 和 LoadStreams，负责读取、缩放及预处理输入。
from utils.utils import *
# utils/utils.py 提供配置解析、类别读取、NMS、坐标缩放和画框等辅助函数。

def detect(save_txt=False, save_img=False):
    # 执行一次完整检测任务，并把结果保存到 opt.output。
    # - save_txt 决定是否把每个检测框另存为文本，
    # - save_img 决定是否保存画框后的图片/视频。 

    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    # ONNX_EXPORT 为 False 时使用用户指定的正方形输入尺寸，例如 416；
    # 本实验不导出 ONNX，所以 img_size 实际等于 opt.img_size。
        
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    # opt 是文件末尾由 `parser.parse_args()` 创建的全局参数对象
    # 一次取出后面频繁使用的命令行参数：输出目录、输入源、权重、是否 FP16、是否显示。
        
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # source 为 摄像头编号 0 / 网络流地址 / .txt 流列表 时，走 LoadStreams
    # 普通图片、文件夹和 MP4/MOV/AVI 则走 LoadImages。本实验输入本地视频，所以结果为 False。
        
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # select_device() 把字符串 "0" 转成 cuda:0，把 "cpu" 转成 CPU 设备。
    # ONNX 导出不属于本作业，所以正常运行时设备由 --device 决定。
        
    # 当前 detect.py 每次运行都清空 --output 目录，避免旧结果与本次结果混在一起：
    # - os.path.exists(out) 判断输出目录是否已经存在；
    # - shutil.rmtree(out) 删除该目录及其内部旧文件；
    # - os.makedirs(out) 重新建立同名空目录。
    # run_exp.sh 为两个模型指定不同目录，防止剪枝模型运行时清除原模型结果。
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    model = Darknet(opt.cfg, img_size) # 根据网络配置创建原模型或剪枝模型
    # cfg 不保存参数，只描述层结构。
    # 原 weights 必须配 yolov3.cfg；剪枝 weights 必须配 prune_0.5_yolov3.cfg，否则通道数不同，参数无法正确加载。
        
    # attempt_download() 在权重不存在时尝试取得已知公开权重；老师提供的本地权重存在时，不需要下载。
    attempt_download(weights)
    # 随后根据扩展名选择对应加载方法。
    if weights.endswith('.pt'):  # pytorch format
        # .pt 是 PyTorch checkpoint；['model'] 取出 state_dict，再复制进网络。
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        # .weights 是 Darknet 二进制格式，必须按层顺序读取 BN/卷积参数。
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # 将模型移动到 CPU/GPU，再切换到 eval 模式。
    # eval() 会让 BatchNorm 使用训练期间保存的 running mean/variance，而不是继续更新统计量。
    model.to(device).eval()
    
    # ONNX 导出分支与本实验无关；开启时用全 0 假输入导出后立即结束检测函数。
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # 只有 CUDA 支持这里的 FP16 路径；CPU 即使收到 --half 也会强制使用 FP32。
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # 普通本地 MP4/MOV/AVI 进入 LoadImages；摄像头、HTTP/RTSP 流进入 LoadStreams。
    vid_path, vid_writer = None, None
    # vid_path 记录当前正在写入的视频路径；vid_writer 保存 OpenCV VideoWriter 对象。
    if webcam:
        view_img = True  # 流输入默认实时显示画面。
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True  # 本地图片/视频默认保存结果，而不要求额外命令行开关。
        dataset = LoadImages(source, img_size=img_size, half=half)

    # parse_data_cfg() 从 coco.data 取 names 路径；load_classes() 按行读出 80 个类别名。
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    # 为每个类别随机生成一种 BGR 颜色；同一次运行中该类别的所有框使用同一颜色。
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # t0 用于统计整个输入处理完成的总墙钟时间，不等于纯模型推理时间。
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # dataset 每次迭代返回：
        # - path：当前图片/视频路径；
        # - img：已完成 letterbox、BGR→RGB、HWC→CHW 和 0~1 归一化的 numpy 数组；
        # - im0s：未缩放的原始 BGR 帧，用于画框和保存；
        # - vid_cap：视频输入对应的 cv2.VideoCapture，图片输入时为 None。
        
        # 每帧开始计时。CUDA 是异步执行的，严格 benchmark 需要 synchronize；
        # 这里同步是为了让写进视频和日志的 FPS 更能反映当前帧实际处理速度。
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.time()

        # numpy 数组转换为 PyTorch tensor，并移动到选定 CPU/GPU。
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            # 单张图的形状是 [C,H,W]；模型需要 [N,C,H,W]，所以在最前面增加 batch 维。
            img = img.unsqueeze(0)
        # 前向传播返回所有候选框。此时还没有去掉置信度低或高度重叠的候选框。
        pred, _ = model(img)

        if opt.half:
            # NMS 使用 FP32 更稳，因此把模型的 FP16 输出转回 float32。
            pred = pred.float()

        # NMS 根据 --conf-thres 过滤低置信度框，再按 --nms-thres 合并重复框。
        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
            if webcam:  # batch_size >= 1
                # 流输入可能一个 batch 含多个流，因此按 i 取得各自路径和原始帧。
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                # 普通文件输入每次只有一张图或一帧。
                p, s, im0 = path, '', im0s

            # 只取输入文件名拼到输出目录，例如 output_original/c_test.mp4。
            save_path = str(Path(out) / Path(p).name)
            # s 是终端结果摘要字符串，先记录模型实际输入的高和宽。
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # 模型框坐标基于 416 输入；scale_coords() 将其映射回原始视频尺寸并取整。
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 按类别统计本帧检测数量，并把例如“2 persons”加入摘要字符串。
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # det 的每一行依次包含 xyxy、object confidence、class confidence、class id。
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        # save_txt 默认 False；开启时将框坐标、类别和置信度追加到文本。
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        # label 例如 person 0.93；plot_one_box() 把标签和彩色边界框画到 im0。
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # CUDA 前向和 NMS 完成后再次同步，避免只测到“提交 GPU 任务”的时间。
            if device.type == 'cuda':
                torch.cuda.synchronize()
            frame_seconds = time.time() - t
            frame_fps = 1.0 / frame_seconds if frame_seconds > 0 else 0.0

            # 把模型名称和逐帧端到端近似 FPS 写到画面左上角。这样合并后的对比视频
            # 不依赖外部日志，也能直观看到两边模型及速度变化。
            overlay = '%s | FPS %.2f' % (opt.model_label, frame_fps)
            cv2.rectangle(im0, (8, 8), (min(im0.shape[1] - 8, 430), 44), (0, 0, 0), -1)
            cv2.putText(im0, overlay, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, (255, 255, 255), 2, cv2.LINE_AA)
            print('%s | FPS ==> (%.3f)' % (opt.model_label, frame_fps))

            # Stream results
            if view_img:
                # --view-img 才弹出窗口；GPU 服务器通常不启用，避免无显示环境报错。
                cv2.imshow(p, im0) #p is path

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    # 单张图片直接使用 cv2.imwrite() 保存。
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        # 第一次遇到某个视频时创建 writer；切换视频前释放上一个 writer。
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        # 输出沿用输入视频的 FPS、宽和高，因此播放时长及画面尺寸不变。
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    # 把当前已画框并叠加模型名称/FPS 的帧写入输出视频。
                    vid_writer.write(im0)

    if save_txt or save_img:
        # os.getcwd() 是当前工作目录；与 out 拼接后打印结果绝对位置。
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            # 仅 macOS 自动用默认应用打开输出；GPU Linux 设备不会执行。
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    # 只有用户直接执行 ``python detect.py ...`` 时才解析参数并开始检测；
    # 如果其它文件 ``import detect``，下面代码不会自动运行。
    parser = argparse.ArgumentParser()
    # add_argument() 声明程序接受的命令行参数；default 是未传参数时的默认值，
    # type 负责把终端文本转换为 Python 类型，help 会显示在 python detect.py --help 中。
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    # --cfg：网络结构文件。原模型和剪枝模型必须使用各自匹配的 cfg。
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    # --data：数据配置；视频检测主要读取其中 names 指向的 COCO 类别名称文件。
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    # --weights：与 cfg 配套的 .weights 或带 ['model'] 字段的 .pt checkpoint。
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    # --source：图片、目录、视频、摄像头编号或网络流地址。
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    # --output：结果目录；detect() 每次运行会先清空它，再保存本次结果。
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    # --img-size：letterbox 后送入模型的正方形边长，不改变输出视频原始分辨率。
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    # --conf-thres：低于该置信度的候选框会被过滤。
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    # --nms-thres：NMS 判断同类别候选框是否高度重叠时使用的 IoU 阈值。
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # --fourcc：OpenCV 写视频时使用的四字符编码名称，默认 mp4v。
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    # action='store_true'：命令行出现 --half 时为 True，未出现时为 False。
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    # --device 0 表示 cuda:0，--device cpu 表示强制使用 CPU。
    parser.add_argument('--view-img', action='store_true', help='display results')
    # --view-img：弹窗实时显示结果；不影响是否保存本地视频。
    parser.add_argument('--model-label', type=str, default='YOLOv3',
                        help='model name drawn on the output video')
    # --model-label：只控制左上角显示文字，不参与模型计算。
    opt = parser.parse_args()
    # parse_args() 真正读取终端参数，得到 opt.cfg、opt.weights、opt.source 等属性。
    print(opt)

    # 推理不需要梯度；no_grad() 可减少显存使用并避免建立反向传播计算图。
    with torch.no_grad():
        detect()
