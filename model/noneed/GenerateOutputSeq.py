from config import IGNORE_ID,VOCAB
import torch
import numpy as np
# def pad_list(x, pad_value):
#     n_batch = len(x)
#     max_len = max(x.size(0) for x in x)
#     pad = x[0].new(n_batch, max_len, *x[0].size()[1:]).fill_(pad_value)
#     for i in range(n_batch):
#         pad[i, :x[i].size(0)] = x[i]
#     return pad
# 进行偏移输入,解码器的每个时间步的输出作为下一个时间步的输入
# 这样才能预测下一个字符编码
def in_out_process(padded_input):
    """Generate decoder input and output label from padded_input
    Add <sos> to decoder input, and add <eos> to decoder output label
    """
    ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
    # prepare input and output word sequences with sos/eos IDs
    sos = ys[0].new([0])
    eos = ys[0].new([1])
    ignore=ys[0].new([IGNORE_ID])
    ys_in = [torch.cat([sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, ignore], dim=0) for y in ys]
    ys_in_pad = pad_list(ys_in, eos[0])
    ys_out_pad = pad_list(ys_out, IGNORE_ID)
    assert ys_in_pad.size() == ys_out_pad.size()
    return ys_in_pad, ys_out_pad