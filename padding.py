import numpy as np
from config import IGNORE_ID
import torch
from torch.utils.data.dataloader import default_collate
# 填充,对patch排序
def pad_collate(batch):
    max_input_len = float('-inf')  # 最长输入序列长度的初始值
    max_target_len = float('-inf')  # 最长目标序列长度的初始值

    # 遍历批量中的每个元素，获取最大的输入序列长度和目标序列长度
    for elem in batch:
        feature, trn = elem
        # 更新最大输入序列长度
        max_input_len = max(max_input_len, feature.shape[0])
        # 更新最大目标序列长度
        max_target_len = max(max_target_len, len(trn))

    # 对每个元素进行填充和处理
    for i, elem in enumerate(batch):
        feature, trn = elem
        input_length = feature.shape[0]  # 当前输入序列的长度
        input_dim = feature.shape[1]  # 输入特征的维度
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)  # 创建全零矩阵作为填充后的输入序列
        padded_input[:input_length, :] = feature  # 将原始特征复制到填充后的输入序列中
        padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)  # 使用常数值（IGNORE_ID）对目标序列进行填充
        batch[i] = (padded_input, padded_target, input_length)  # 更新批量中的元素为填充后的输入、目标序列以及输入序列长度

    # 根据输入序列长度对批量进行排序（从长到短）
    batch.sort(key=lambda x: x[2], reverse=True)

    # return default_collate(batch)
    return default_collate([(torch.from_numpy(padded_input), torch.from_numpy(padded_target), input_length) for padded_input, padded_target, input_length in batch])
