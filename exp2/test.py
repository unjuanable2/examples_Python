"""使用训练好的 AlexNet 权重，对一张外部真实图片进行分类推理。"""

import argparse
# argparse 是 Python 标准库，用来读取命令行参数。
# 例如 run_test.sh 会通过 --image、--weights-dir 和 --output 传入路径。

from pathlib import Path
# Path 用对象的方式表示和拼接文件路径，比手动拼接字符串更安全。

import torch
# torch 用来加载 checkpoint、选择 GPU、执行模型推理并计算分类概率。

from PIL import Image, ImageDraw, ImageFont
# PIL 负责读取测试图片、在图片上绘制预测结果以及保存结果图片。

from torchvision import transforms
# transforms 用来把普通图片转换为 AlexNet 可以接收的张量。

from models.alexnet import AlexNet
# 导入 exp2 训练时使用的 AlexNet 模型结构。
# 推理时必须先创建相同的模型结构，才能正确加载训练保存的参数。


##############################################################################
# CIFAR-10 类别名称                                                         #
##############################################################################
CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
# AlexNet 最后一层输出 10 个分数，它们的下标 0～9 按顺序对应以上类别。


##############################################################################
# 单张测试图片的预处理                                                       #
##############################################################################
PREPROCESS = transforms.Compose([
    transforms.Resize(
        (32, 32),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    # CIFAR-10 图片尺寸是 32×32，因此外部图片也必须缩放到相同尺寸。
    # BICUBIC 表示使用双三次插值，缩放效果通常比最邻近插值平滑。

    transforms.ToTensor(),
    # 将 PIL 图片转换为 PyTorch 张量：
    # 图片形状由 [高度, 宽度, 通道] 变为 [通道, 高度, 宽度]，
    # 像素值也会从 0～255 转换到 0～1。

    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010],
    ),
    # 使用与 exp2 训练、测试阶段相同的 CIFAR-10 均值和标准差。
    # 推理预处理必须与训练预处理一致，否则输入数据分布会发生变化。
])


def load_checkpoint(path, map_location="cpu"):
    """兼容新旧 PyTorch 版本，将一个 checkpoint 加载到 CPU。"""
    try:
        # 新版 PyTorch 支持 weights_only=True，可限制反序列化的对象类型。
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 较旧版本没有 weights_only 参数，出现 TypeError 时使用旧接口加载。
        return torch.load(path, map_location=map_location)


def find_best_checkpoint(weights_dir):
    """检查权重目录，返回其中测试准确率最高的有效 checkpoint。"""
    candidates = sorted(weights_dir.glob("*.pt"))
    # glob("*.pt") 找到 weights/alexnet 目录下的所有 .pt 文件。
    # sorted() 固定检查顺序，使每次运行的输出顺序一致。

    if not candidates:
        # 没有权重时直接停止，避免使用未训练的随机模型进行推理。
        raise FileNotFoundError(f"没有在该目录找到 .pt 权重：{weights_dir}")

    valid = []
    # valid 用来保存所有成功读取的权重信息。
    # 每项内容依次是：测试准确率、epoch、路径、完整 checkpoint。

    for path in candidates:
        try:
            checkpoint = load_checkpoint(path)

            # train.py 保存的 checkpoint 包含 net、acc 和 epoch 三项。
            accuracy = float(checkpoint["acc"])
            epoch = int(checkpoint["epoch"])

            if "net" not in checkpoint:
                # net 保存真正的模型参数；没有 net 的文件不能用于推理。
                raise KeyError("net")

            valid.append((accuracy, epoch, path, checkpoint))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            # 某个文件损坏或格式不正确时跳过它，继续检查其他权重。
            print(f"跳过无效权重 {path.name}：{error}")

    if not valid:
        raise RuntimeError(f"该目录中没有有效的 AlexNet 权重：{weights_dir}")

    # max() 首先比较测试准确率 accuracy。
    # 如果两个权重的准确率完全相同，则选择 epoch 更大的那个。
    return max(valid, key=lambda item: (item[0], item[1]))


def choose_font(size=30):
    """选择常见的粗体字体；系统没有这些字体时使用 Pillow 默认字体。"""
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            # 当前字体不存在时继续尝试列表中的下一个字体。
            pass

    return ImageFont.load_default()


def draw_prediction(image, text):
    """在图片左上角绘制预测文字，并返回一张新的结果图片。"""
    result = image.copy()
    # 使用副本绘制，避免修改内存中的原始推理图片。

    draw = ImageDraw.Draw(result)
    font = choose_font()

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    # textbbox() 计算文字需要占用的矩形区域，以便确定背景框大小。

    padding = 8
    box = (0, 0, right - left + padding * 2, bottom - top + padding * 2)
    draw.rectangle(box, fill=(0, 0, 0))
    # 在文字下面绘制黑色背景，避免文字与照片背景混在一起。

    draw.text(
        (padding, padding - top),
        text,
        font=font,
        fill=(255, 255, 0),
    )
    # 使用黄色写出类似 bird (92.15%) 的类别和置信度。

    return result


##############################################################################
# 解析命令行参数                                                             #
##############################################################################
def parse_args():
    script_dir = Path(__file__).resolve().parent
    # __file__ 是当前 test.py 的路径，parent 得到 exp2 的绝对路径。
    # 使用绝对路径后，即使从其他目录运行脚本，也能找到图片和权重。

    parser = argparse.ArgumentParser(description="AlexNet 单张图片推理")

    parser.add_argument(
        "--image",
        type=Path,
        default=script_dir / "test.jpeg",
        help="需要进行分类的输入图片路径",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "test_result.jpeg",
        help="写有预测结果的输出图片路径",
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=script_dir / "weights" / "alexnet",
        help="保存 AlexNet .pt 权重的目录",
    )

    return parser.parse_args()


##############################################################################
# 单张图片推理主流程                                                         #
##############################################################################
def main():
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"找不到输入图片：{args.image}")

    accuracy, epoch, checkpoint_path, checkpoint = find_best_checkpoint(
        args.weights_dir
    )
    # 自动选择 checkpoint 内 acc 最大的权重。
    # 对本实验的完整权重目录，预期选择测试准确率 83.050%、epoch 131 的权重。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CUDA 可用时使用 NVIDIA GPU，否则使用 CPU。
    # run_test.sh 已经强制检查 CUDA，所以通过脚本运行时这里应当是 cuda。

    model = AlexNet().to(device)
    # 创建与训练阶段完全相同的 AlexNet，并把它移动到推理设备。

    model.load_state_dict(checkpoint["net"])
    # 将最佳 checkpoint 中保存的参数加载到模型。

    model.eval()
    # 切换到评估模式，关闭 Dropout 的随机丢弃行为。
    # 如果不调用 eval()，同一张图片多次推理可能得到不同结果。

    with Image.open(args.image) as opened_image:
        image = opened_image.convert("RGB")
    # convert("RGB") 保证输入一定有三个颜色通道，同时兼容灰度图和 RGBA 图片。

    input_tensor = PREPROCESS(image).unsqueeze(0).to(device)
    # PREPROCESS 输出形状 [3, 32, 32]。
    # unsqueeze(0) 在最前面添加 batch 维，得到 [1, 3, 32, 32]。

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    # inference_mode() 表示这里只进行推理，不记录梯度，可以减少显存占用。
    # 模型输出 logits；softmax 将 10 个原始分数转换为概率。
    # [0] 取出这个 batch 中唯一一张图片的概率向量。

    confidence, class_index = probabilities.max(dim=0)
    # max(dim=0) 返回最大概率以及它在 10 个类别中的下标。

    label = CLASSES[class_index.item()]
    prediction_text = f"{label} ({confidence.item() * 100:.2f}%)"
    # 例如类别下标为 2、概率为 0.9215 时，文字是 bird (92.15%)。

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 输出目录不存在时自动创建；目录已经存在时不会报错。

    result = draw_prediction(image, prediction_text)
    result.save(args.output, quality=95)
    # 将带有预测文字的图片另存为 test_result.jpeg，不覆盖 test.jpeg 原图。

    print(f"推理设备：{device}")
    print(f"选择的权重：{checkpoint_path}")
    print(f"权重测试准确率：{accuracy:.3f}%（epoch {epoch + 1}）")
    print(f"图片预测结果：{prediction_text}")
    print(f"结果图片路径：{args.output}")


if __name__ == "__main__":
    # 直接执行 python test.py 时调用 main()；被其他模块导入时不会自动运行。
    main()
