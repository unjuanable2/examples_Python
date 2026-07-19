import csv
# csv 是 Python 标准库，用来写 .csv 表格文件。

import re
# re 是 Python 标准库，用来做正则匹配。
# 这里用它从 results_analysis/run_exp2_out.txt 里提取 Epoch、Learning Rate、Loss、Acc 等字段。

from pathlib import Path
# pathlib.Path 用来处理文件路径，比直接写字符串更清楚。


RESULTS_DIR = Path(__file__).resolve().parent
# __file__ 是当前这个 Python 脚本文件自己的路径。
# Path(__file__).resolve().parent 表示：
# - 先找到 analyze_exp2_log.py 的绝对路径；
# - 再取出它所在的文件夹。
#
# 因为这个脚本本身放在 results_analysis/ 里，
# 所以 RESULTS_DIR 就是 results_analysis/ 的绝对路径。
#
# 这样 exp1 根目录里只放代码和 README，
# 训练日志、CSV、曲线图等结果文件都集中放到这个文件夹里。

LOG_FILE = RESULTS_DIR / "run_exp2_out.txt"
# 要分析的训练日志文件。
# / 是 pathlib.Path 的路径拼接写法。
# 这里表示 “脚本所在文件夹/run_exp2_out.txt”，也就是 results_analysis/run_exp2_out.txt。

CSV_FILE = RESULTS_DIR / "exp2_epoch_metrics.csv"
# 输出的 CSV 表格文件。
# CSV 可以用 Excel、Numbers、WPS、Python 等工具打开。

ACCURACY_FIG = RESULTS_DIR / "exp2_accuracy_curve.png"
# 输出的准确率曲线图片。

LOSS_FIG = RESULTS_DIR / "exp2_loss_curve.png"
# 输出的 loss 曲线图片。

ACCURACY_SVG = RESULTS_DIR / "exp2_accuracy_curve.svg"
# 如果当前 Python 环境没有 matplotlib，就输出 SVG 版本的准确率曲线。

LOSS_SVG = RESULTS_DIR / "exp2_loss_curve.svg"
# 如果当前 Python 环境没有 matplotlib，就输出 SVG 版本的 loss 曲线。


def parse_log(log_text):
    # def parse_log(...) 定义一个函数。
    #
    # 参数 log_text：
    # - 是 results_analysis/run_exp2_out.txt 的完整文本内容；
    # - 这个函数会从里面提取每个 epoch 的训练结果和测试结果。

    epoch_blocks = re.split(r"(?=Epoch:\s*\d+)", log_text)
    # re.split(...) 按 "Epoch: 数字" 切分日志。
    #
    # (?=...) 是正则里的“向前看”写法：
    # - 它只负责找到切分位置；
    # - 不会把 "Epoch: 1" 这几个字删掉。
    #
    # 切分后，每个 block 大致对应一个 epoch。

    rows = []
    # rows 是列表，用来保存解析出的每个 epoch 的结果。

    for block in epoch_blocks:
        # 遍历每个 epoch 的日志块。

        epoch_match = re.search(r"Epoch:\s*(\d+)", block)
        # 找当前 block 里的 epoch 编号。

        if not epoch_match:
            # 如果这个 block 里没有 Epoch，就跳过。
            # 日志开头的模型信息不是某个 epoch，所以会走到这里。
            continue

        epoch = int(epoch_match.group(1))
        # group(1) 取出正则里第一个括号匹配到的内容，也就是 epoch 数字。
        # int(...) 把字符串转成整数。

        lr_match = re.search(r"Learning Rate:\s*([^\s]+)", block)
        # 找当前 epoch 的学习率。

        learning_rate = lr_match.group(1) if lr_match else ""
        # 如果找到了学习率，就保存；如果没找到，就用空字符串。

        progress_items = []
        # progress_items 保存当前 epoch 里所有进度条记录。
        # 每个 batch 都会打印一次 Loss/Acc，所以这里会有很多条。

        pattern = (
            r"Loss:\s*([0-9.]+)\s*\|\s*Acc:\s*([0-9.]+)%\s*"
            r"\((\d+)/(\d+)\).*?(\d+)/(\d+)"
        )
        # 这个正则用来匹配类似：
        # Loss: 2.319 | Acc: 13.536% (6768/50000) 391/391
        #
        # 括号里会提取出：
        # 1. loss
        # 2. acc 百分比
        # 3. correct 正确样本数
        # 4. samples 已统计样本数
        # 5. 当前 batch 编号
        # 6. 总 batch 数

        for match in re.finditer(pattern, block):
            # re.finditer(...) 会找出当前 block 里所有匹配项。

            progress_items.append({
                "loss": float(match.group(1)),
                "acc": float(match.group(2)),
                "correct": int(match.group(3)),
                "samples": int(match.group(4)),
                "batch": int(match.group(5)),
                "batches": int(match.group(6)),
            })
            # 把每条进度记录保存成一个字典 dict。

        train_item = None
        test_item = None
        # train_item 保存当前 epoch 训练阶段最后一条记录。
        # test_item 保存当前 epoch 测试阶段最后一条记录。

        for item in reversed(progress_items):
            # reversed(...) 从后往前找。
            # 因为一个阶段最后一条记录才是“这一整个阶段”的最终统计结果。

            if train_item is None and (item["samples"] == 50000 or item["batches"] == 391):
                # CIFAR-10 训练集有 50000 张图。
                # batch_size=128 时，一个 epoch 有 391 个 batch。
                train_item = item

            if test_item is None and (item["samples"] == 10000 or item["batches"] == 79):
                # CIFAR-10 测试集有 10000 张图。
                # batch_size=128 时，测试阶段有 79 个 batch。
                test_item = item

            if train_item is not None and test_item is not None:
                # 如果训练结果和测试结果都找到了，就不用继续找了。
                break

        rows.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_item["loss"] if train_item else "",
            "train_acc": train_item["acc"] if train_item else "",
            "train_correct": train_item["correct"] if train_item else "",
            "train_samples": train_item["samples"] if train_item else "",
            "test_loss": test_item["loss"] if test_item else "",
            "test_acc": test_item["acc"] if test_item else "",
            "test_correct": test_item["correct"] if test_item else "",
            "test_samples": test_item["samples"] if test_item else "",
        })
        # 每个 epoch 最终保存一行。

    return rows
    # return 表示把解析结果返回给调用者。


def write_csv(rows):
    # 把解析出的结果写入 CSV 文件。

    RESULTS_DIR.mkdir(exist_ok=True)
    # mkdir(...) 创建 results_analysis 文件夹。
    # exist_ok=True 表示如果文件夹已经存在，也不要报错。

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_acc",
        "train_correct",
        "train_samples",
        "test_loss",
        "test_acc",
        "test_correct",
        "test_samples",
    ]
    # CSV 的列名。

    with CSV_FILE.open("w", newline="") as f:
        # with ... as ... 是上下文管理写法。
        # 文件写完后会自动关闭。

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # DictWriter 可以把字典列表写成 CSV。

        writer.writeheader()
        # 写第一行表头。

        writer.writerows(rows)
        # 写所有 epoch 结果。


def write_figures(rows):
    # 根据 CSV 数据画两张图：
    # - train/test accuracy 曲线；
    # - train/test loss 曲线。
    #
    # 这个函数无论如何都会生成 SVG。
    # 如果当前环境安装了 matplotlib，还会额外生成 PNG。

    write_svg_figures(rows)
    # 先用纯 Python 生成 SVG，保证 README 引用的图片每次都会更新。

    try:
        import matplotlib
        matplotlib.use("Agg")
        # Agg 是非交互式后端。
        # 这样脚本在没有图形界面的 Linux 服务器上也能保存图片。

        import matplotlib.pyplot as plt
    except ImportError:
        # 如果虚拟环境里没有 matplotlib，就只保留前面生成的 SVG。
        print("matplotlib not found; SVG figures were generated, PNG figures were skipped.")
        return

    epochs = [row["epoch"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]
    test_acc = [row["test_acc"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    test_loss = [row["test_loss"] for row in rows]

    plt.figure()
    # 新建第一张图。

    plt.plot(epochs, train_acc, label="train accuracy")
    plt.plot(epochs, test_acc, label="test accuracy")
    # 画训练准确率和测试准确率。

    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.title("Exp1 Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(ACCURACY_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    # 保存准确率曲线并关闭图像。

    plt.figure()
    # 新建第二张图。

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    # 画训练 loss 和测试 loss。

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Exp1 Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    # 保存 loss 曲线并关闭图像。


def _scale_points(xs, ys, width, height, margin, y_min=None, y_max=None):
    # 这个内部辅助函数用来把真实数据点转换成 SVG 画布上的坐标点。
    #
    # 真实数据：
    # - x 是 epoch，例如 1 到 200；
    # - y 是 accuracy 或 loss。
    #
    # SVG 坐标：
    # - 左上角是 (0, 0)；
    # - x 往右变大；
    # - y 往下变大。

    x_min = min(xs)
    x_max = max(xs)
    # x 轴范围。

    if y_min is None:
        y_min = min(ys)
    if y_max is None:
        y_max = max(ys)
    # y 轴范围。

    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    # 防止分母为 0。

    points = []
    for x, y in zip(xs, ys):
        px = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
        # 把 epoch 映射到 SVG 横坐标。

        py = height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)
        # 把数值映射到 SVG 纵坐标。
        # 注意 SVG 的 y 轴向下，所以这里要用 height - margin - ...

        points.append(f"{px:.2f},{py:.2f}")

    return " ".join(points)


def _write_svg(path, title, y_label, epochs, series):
    # 用纯文本写一个简单 SVG 折线图。
    #
    # 参数 series 是一个列表，里面每一项是：
    # (曲线名称, y_values, 颜色)

    width = 900
    height = 520
    margin = 70

    all_y = []
    for _, ys, _ in series:
        all_y.extend(ys)

    y_min = min(all_y)
    y_max = max(all_y)
    padding = (y_max - y_min) * 0.08 if y_max != y_min else 1
    y_min -= padding
    y_max += padding
    # 给 y 轴上下留一点空白，避免曲线贴边。

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial">epoch</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {height / 2})">{y_label}</text>',
        f'<text x="{margin}" y="{height - margin + 24}" text-anchor="middle" font-size="12" font-family="Arial">{min(epochs)}</text>',
        f'<text x="{width - margin}" y="{height - margin + 24}" text-anchor="middle" font-size="12" font-family="Arial">{max(epochs)}</text>',
        f'<text x="{margin - 10}" y="{height - margin + 4}" text-anchor="end" font-size="12" font-family="Arial">{y_min:.3g}</text>',
        f'<text x="{margin - 10}" y="{margin + 4}" text-anchor="end" font-size="12" font-family="Arial">{y_max:.3g}</text>',
    ]

    legend_y = 60
    for name, ys, color in series:
        points = _scale_points(epochs, ys, width, height, margin, y_min, y_max)
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>'
        )

        elements.append(
            f'<line x1="{width - 220}" y1="{legend_y}" x2="{width - 185}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{width - 175}" y="{legend_y + 5}" font-size="13" font-family="Arial">{name}</text>'
        )
        legend_y += 22

    elements.append("</svg>")
    path.write_text("\n".join(elements))


def write_svg_figures(rows):
    # 没有 matplotlib 时，用这个函数生成两张 SVG 曲线图。

    epochs = [row["epoch"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]
    test_acc = [row["test_acc"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    test_loss = [row["test_loss"] for row in rows]

    _write_svg(
        ACCURACY_SVG,
        "Exp1 Accuracy Curve",
        "accuracy (%)",
        epochs,
        [
            ("train accuracy", train_acc, "#1f77b4"),
            ("test accuracy", test_acc, "#d62728"),
        ],
    )

    _write_svg(
        LOSS_SVG,
        "Exp1 Loss Curve",
        "loss",
        epochs,
        [
            ("train loss", train_loss, "#1f77b4"),
            ("test loss", test_loss, "#d62728"),
        ],
    )


def print_summary(rows):
    # 在终端打印一个简短总结，方便运行脚本后快速看结果。

    if not rows:
        print("No epoch metrics found.")
        return

    rows_with_test = [row for row in rows if row["test_acc"] != ""]
    rows_with_train = [row for row in rows if row["train_acc"] != ""]

    best_test = max(rows_with_test, key=lambda row: row["test_acc"]) if rows_with_test else None
    best_train = max(rows_with_train, key=lambda row: row["train_acc"]) if rows_with_train else None
    last = rows[-1]

    print(f"Parsed epochs: {len(rows)}")
    print(f"Wrote CSV: {CSV_FILE}")

    if best_test:
        print(
            "Best test accuracy: "
            f"{best_test['test_acc']:.3f}% at epoch {best_test['epoch']}"
        )

    if best_train:
        print(
            "Best train accuracy: "
            f"{best_train['train_acc']:.3f}% at epoch {best_train['epoch']}"
        )

    print(
        "Last epoch: "
        f"epoch {last['epoch']}, "
        f"train_acc={last['train_acc']:.3f}%, "
        f"test_acc={last['test_acc']:.3f}%"
    )


def main():
    # main() 是脚本入口函数。

    RESULTS_DIR.mkdir(exist_ok=True)
    # 确保 results_analysis 文件夹存在。

    if not LOG_FILE.exists():
        # 如果日志文件不存在，直接报错退出。
        raise FileNotFoundError(f"Log file not found: {LOG_FILE}")

    log_text = LOG_FILE.read_text(errors="replace")
    # 读取日志文本。
    # errors="replace" 表示遇到无法解码的字符时，用替代字符处理，不让脚本直接崩掉。

    rows = parse_log(log_text)
    # 解析每个 epoch 的训练/测试结果。

    write_csv(rows)
    # 写 CSV 表格。

    write_figures(rows)
    # 写两张曲线图。

    print_summary(rows)
    # 打印简短总结。


if __name__ == "__main__":
    # 只有直接运行 python results_analysis/analyze_exp2_log.py 时，才执行 main()。
    # 如果以后别的文件 import 这个脚本，不会自动开始分析。
    main()
