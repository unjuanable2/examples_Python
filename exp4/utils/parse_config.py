"""解析 Darknet 的 cfg 网络文件和 ``coco.data`` 数据配置。

cfg 中每个 ``[convolutional]``/``[shortcut]``/``[yolo]`` 块会被转成一个字典；
models.py 再把字典变成 PyTorch 层。coco.data 则提供类别数、5k.txt 和 coco.names。
"""

import numpy as np
import torch

def parse_model_cfg(path):
    """读取 cfg 文本，返回保持原顺序的网络层配置字典列表。"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            else:
                mdefs[-1][key] = val.strip()

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

