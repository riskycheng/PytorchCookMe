import random
from IPython import display
from matplotlib import pyplot as plt
import torch


# 打印散点图
def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figureSize(figsize=(10, 6)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)
