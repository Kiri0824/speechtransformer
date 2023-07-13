import torch
import torch.nn as nn
from config import device
class PositionalEncoding(nn.Module):
    # PE(pos,2i) = sin(pos/10000^(2i/dmodel))
    # PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
    def __init__(self, d_model, max_len, device=device):
        super(PositionalEncoding, self).__init__()
        # max_len,d_model大小全0矩阵
        encoding = torch.zeros(max_len, d_model, device=device)
        encoding.requires_grad = False  
        # 所有位置0~max_len
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 偶数索引
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 偶数
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        # 奇数
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding)
    def forward(self,x):
        # x:torch.Size([32, 270, 320])
        seq_len= x.size(1)
        return self.encoding[:, :seq_len,:]
