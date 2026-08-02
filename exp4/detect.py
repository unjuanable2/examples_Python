"""使用 YOLOv3 对图片或视频做目标检测。

本文件是 exp4 最终成果的推理入口。程序按以下顺序工作：

1. argparse 读取 cfg、weights、视频、阈值和设备等命令行参数；
2. Darknet(cfg) 根据网络配置创建原模型或剪枝模型；
3. load_darknet_weights() 把 .weights 参数加载到对应网络；
4. LoadImages 逐帧读取图片/视频并执行 letterbox、BGR→RGB 和 0~1 归一化；
5. model(img) 前向传播，non_max_suppression() 删除重复检测框；
6. scale_coords() 将 416×416 输入坐标映射回原视频，plot_one_box() 画框；
7. OpenCV VideoWriter 保存带检测框、模型名称和当前 FPS 的结果视频。

原模型和剪枝模型共用本文件，区别只来自 --cfg 与 --weights 的正确配对。
"""

import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

def detect(save_txt=False, save_img=False):
    """执行一次完整检测任务，并把结果保存到 ``opt.output``。

    ``opt`` 是文件末尾通过 ``parser.parse_args()`` 创建的全局命令行参数对象。
    旧项目采用这种写法；较新的工程通常会把 opt 显式作为函数参数传入。
    """
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # select_device() 把字符串 "0" 转成 cuda:0，把 "cpu" 转成 CPU 设备。
    # ONNX 导出不属于本作业，所以正常运行时设备由 --device 决定。
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # 注意：旧版 detect.py 会删除整个输出目录再重建，所以 run_exp.sh 为两个模型
    # 分别指定 output_original 和 output_pruned50，防止后一次覆盖前一次。
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # cfg 不保存参数，只描述层结构。原 weights 必须配 yolov3.cfg；剪枝 weights
    # 必须配 prune_0.5_yolov3.cfg，否则通道数不同，参数无法正确加载。
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # 将模型移动到 CPU/GPU，再切换到 eval 模式。eval() 会让 BatchNorm 使用训练期间
    # 保存的 running mean/variance，而不是继续更新统计量。
    model.to(device).eval()
    
    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # 普通本地 MP4/MOV/AVI 进入 LoadImages；摄像头、HTTP/RTSP 流进入 LoadStreams。
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # t0 用于统计整个输入处理完成的总墙钟时间，不等于纯模型推理时间。
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # 每帧开始计时。CUDA 是异步执行的，严格 benchmark 需要 synchronize；
        # 这里同步是为了让写进视频和日志的 FPS 更能反映当前帧实际处理速度。
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # 前向传播返回所有候选框。此时还没有去掉置信度低或高度重叠的候选框。
        pred, _ = model(img)

        if opt.half:
            pred = pred.float()

        # NMS 根据 --conf-thres 过滤低置信度框，再按 --nms-thres 合并重复框。
        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
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
                cv2.imshow(p, im0) #p is path

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    # 只有用户直接执行 ``python detect.py ...`` 时才解析参数并开始检测；
    # 如果其它文件 ``import detect``，下面代码不会自动运行。
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--model-label', type=str, default='YOLOv3',
                        help='model name drawn on the output video')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
