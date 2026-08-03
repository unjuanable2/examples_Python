#!/usr/bin/env python3
"""分析 exp4 日志，并把报告同时输出到终端和文本文件。

运行方式（当前目录可以是 exp4，也可以是 results_analysis）：

    python results_analysis/analysis_exp4.py
    python analysis_exp4.py

脚本只使用 Python 标准库，不需要 PyTorch、OpenCV 或 NumPy。
"""

import re
import sys
from pathlib import Path


# __file__ 是本脚本自身的路径，因此无论从哪个目录启动，都能找到同目录日志。
RESULTS_DIR = Path(__file__).resolve().parent
REPORT_FILE = RESULTS_DIR / "analysis_exp4_log.txt"


class Tee:
    """把 print() 的内容同时写到终端和报告文件。"""

    def __init__(self, terminal, report):
        self.terminal = terminal
        self.report = report

    def write(self, text):
        self.terminal.write(text)
        self.report.write(text)

    def flush(self):
        self.terminal.flush()
        self.report.flush()


def read_log(name):
    """读取日志；缺失时抛出带文件名的错误。"""
    path = RESULTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少日志：{path}")
    # errors='replace' 可避免日志中偶然出现异常字节时整个分析中断。
    return path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")


def one_match(pattern, text, description, flags=0):
    """取得一个正则结果；没有找到时明确指出缺少哪项数据。"""
    match = re.search(pattern, text, flags)
    if not match:
        raise ValueError(f"日志中没有找到：{description}")
    return match


def parse_detection_log(text):
    """从 detect.py 日志提取帧数、逐帧 FPS 和整个阶段耗时。"""
    fps_values = [float(value) for value in re.findall(r"FPS ==> \(([0-9.]+)\)", text)]
    if not fps_values:
        raise ValueError("检测日志中没有找到逐帧 FPS")

    done = one_match(r"Done\. \(([0-9.]+)s\)", text, "Done 总耗时")
    frame_records = re.findall(r"\((\d+)/(\d+)\).*?FPS ==>", text)
    processed_frames = int(frame_records[-1][0]) if frame_records else len(fps_values)
    total_frames = int(frame_records[-1][1]) if frame_records else len(fps_values)

    return {
        "fps_values": fps_values,
        "average_fps": sum(fps_values) / len(fps_values),
        "minimum_fps": min(fps_values),
        "maximum_fps": max(fps_values),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "total_seconds": float(done.group(1)),
    }


def parse_prune_log(text):
    """从 shortcut_prune.py 日志提取剪枝、COCO 和计时结果。"""
    threshold = one_match(
        r"Gamma value less than ([0-9.eE+-]+) are pruned", text, "BN gamma 阈值"
    )
    channels = one_match(
        r"channels has been reduced from (\d+) to (\d+)",
        text,
        "剪枝前后通道数",
        re.IGNORECASE,
    )

    # 日志中通常有两次 all：第一次是 mask 置零的大模型，第二次是 compact model。
    coco_rows = re.findall(
        r"^\s*all\s+([0-9.eE+]+)\s+([0-9.eE+]+)\s+"
        r"([0-9.eE+]+)\s+([0-9.eE+]+)\s+([0-9.eE+]+)\s+([0-9.eE+]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not coco_rows:
        raise ValueError("剪枝日志中没有找到 COCO all 汇总行")

    table_map = one_match(r"\| mAP\s+\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|", text, "mAP 表")
    table_params = one_match(
        r"\| Parameters\s+\|\s*(\d+)\s*\|\s*(\d+)\s*\|", text, "参数量表"
    )
    table_inference = one_match(
        r"\| Inference\s+\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|", text, "推理时间表"
    )

    return {
        "threshold": float(threshold.group(1)),
        "channels_before": int(channels.group(1)),
        "channels_after": int(channels.group(2)),
        "coco_rows": [tuple(float(value) for value in row) for row in coco_rows],
        "reference_map_before": float(table_map.group(1)),
        "measured_map_after": float(table_map.group(2)),
        "parameters_before": int(table_params.group(1)),
        "parameters_after": int(table_params.group(2)),
        "inference_before": float(table_inference.group(1)),
        "inference_after": float(table_inference.group(2)),
        "cfg_saved": "Config file has been saved:" in text,
        "weights_saved": "Compact model has been saved:" in text,
    }


def percent_reduction(before, after):
    return (before - after) / before * 100.0


def file_size_mib(path):
    return path.stat().st_size / 1024.0 / 1024.0 if path.is_file() else None


def format_size(path):
    size = file_size_mib(path)
    return f"{size:.2f} MiB" if size is not None else "文件不存在"


def print_report(original, pruned_video, pruning):
    """计算派生指标并打印最终中文报告。"""
    channel_reduction = percent_reduction(pruning["channels_before"], pruning["channels_after"])
    parameter_reduction = percent_reduction(pruning["parameters_before"], pruning["parameters_after"])
    parameter_ratio = pruning["parameters_before"] / pruning["parameters_after"]
    inference_reduction = percent_reduction(pruning["inference_before"], pruning["inference_after"])
    inference_speedup = pruning["inference_before"] / pruning["inference_after"]
    fps_gain = (pruned_video["average_fps"] / original["average_fps"] - 1.0) * 100.0
    fps_ratio = pruned_video["average_fps"] / original["average_fps"]
    runtime_reduction = percent_reduction(original["total_seconds"], pruned_video["total_seconds"])
    runtime_speedup = original["total_seconds"] / pruned_video["total_seconds"]

    original_video = RESULTS_DIR / "output_original" / "c_test.mp4"
    pruned_output_video = RESULTS_DIR / "output_pruned50" / "c_test.mp4"
    comparison_video = RESULTS_DIR / "comparison_prune50.mp4"

    print("=" * 68)
    print("exp4：YOLOv3 50% 阈值结构化通道剪枝结果分析")
    print("=" * 68)
    print()
    print("[1] 文件完整性")
    print(f"原模型日志帧数       : {original['processed_frames']}/{original['total_frames']}")
    print(f"剪枝模型日志帧数     : {pruned_video['processed_frames']}/{pruned_video['total_frames']}")
    print(f"原模型检测视频       : {format_size(original_video)}")
    print(f"剪枝模型检测视频     : {format_size(pruned_output_video)}")
    print(f"左右对比视频         : {format_size(comparison_video)}")
    print(f"剪枝 cfg 保存记录    : {'有' if pruning['cfg_saved'] else '无'}")
    print(f"剪枝 weights 保存记录: {'有' if pruning['weights_saved'] else '无'}")
    print()

    print("[2] 通道与参数量")
    print(f"BN gamma 全局阈值    : {pruning['threshold']:.10f}")
    print(
        f"可剪枝通道           : {pruning['channels_before']} -> "
        f"{pruning['channels_after']}（减少 {channel_reduction:.2f}%）"
    )
    print(
        f"模型参数量           : {pruning['parameters_before']:,} -> "
        f"{pruning['parameters_after']:,}（减少 {parameter_reduction:.2f}%，"
        f"约压缩 {parameter_ratio:.2f} 倍）"
    )
    print("说明                 : percent=0.5 是 gamma 排序阈值位置；受 shortcut")
    print("                       共享 mask 约束影响，实际通道减少率不必恰好为 50%。")
    print()

    print("[3] COCO 5k 验证")
    for index, row in enumerate(pruning["coco_rows"], start=1):
        images, targets, precision, recall, map50, f1 = row
        model_name = "mask 置零模型" if index == 1 else "compact model"
        print(
            f"第 {index} 次（{model_name}）: images={images:.0f}, targets={targets:.0f}, "
            f"P={precision:.4f}, R={recall:.4f}, mAP@0.5={map50:.4f}, F1={f1:.4f}"
        )
    print(f"结果表中的 Before mAP: {pruning['reference_map_before']:.3f}（代码硬编码参考值）")
    print(f"本次 compact 实测 mAP: {pruning['measured_map_after']:.6f}")
    print("注意                 : 0.481 与 0.510488 不是同一次、同条件实测，")
    print("                       不能据此声称剪枝使精度提高。")
    print()

    print("[4] 速度")
    print(
        f"平均逐帧 FPS         : {original['average_fps']:.3f} -> "
        f"{pruned_video['average_fps']:.3f}（提高 {fps_gain:.2f}%，{fps_ratio:.2f} 倍）"
    )
    print(
        f"视频阶段总耗时       : {original['total_seconds']:.3f}s -> "
        f"{pruned_video['total_seconds']:.3f}s（减少 {runtime_reduction:.2f}%，"
        f"约快 {runtime_speedup:.2f} 倍）"
    )
    print(
        f"随机输入前向时间     : {pruning['inference_before']:.4f}s -> "
        f"{pruning['inference_after']:.4f}s（减少 {inference_reduction:.2f}%，"
        f"约快 {inference_speedup:.2f} 倍）"
    )
    print("说明                 : FPS、视频总耗时和随机输入前向时间的计时范围不同，")
    print("                       三组加速倍数不同属于正常现象。")
    print()

    print("[5] 自动结论")
    complete = (
        original["processed_frames"] == original["total_frames"]
        and pruned_video["processed_frames"] == pruned_video["total_frames"]
        and pruning["cfg_saved"]
        and pruning["weights_saved"]
        and len(pruning["coco_rows"]) >= 2
    )
    if complete and parameter_reduction > 0 and fps_gain > 0:
        print("结果正常：两段检测均完整结束，mask 模型和 compact model 均完成 COCO")
        print("5k 验证，剪枝模型参数量明显减少，同一视频的平均 FPS 有提升。")
    else:
        print("结果需要检查：存在输出不完整、参数量未下降或平均 FPS 未提升的情况。")
    print("日志不能判断具体哪一帧发生漏检或误检；这需要观看左右对比视频。")
    print()
    print(f"本报告已保存到：{REPORT_FILE}")


def main():
    original_text = read_log("detect_original.log")
    pruned_video_text = read_log("detect_pruned50.log")
    prune_text = read_log("prune50.log")

    original = parse_detection_log(original_text)
    pruned_video = parse_detection_log(pruned_video_text)
    pruning = parse_prune_log(prune_text)
    print_report(original, pruned_video, pruning)


if __name__ == "__main__":
    # 用 with 自动关闭报告文件；Tee 保证终端与文件内容完全一致。
    with REPORT_FILE.open("w", encoding="utf-8") as report:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        tee = Tee(original_stdout, report)
        sys.stdout = tee
        sys.stderr = tee
        try:
            main()
        except Exception as error:
            print(f"分析失败：{error}")
            raise SystemExit(1)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
