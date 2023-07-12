import torch
import torch.nn as nn
from .InputEncoding import InputEncoding
from .PositionalEncoding import PositionalEncoding
from config import device
class Encoder(nn.Module):
    def __init__(self, d_input=159,n_layers=6, n_head=8, 
                 d_model=512, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.encoderlayers=nn.ModuleList([nn.TransformerEncoderLayer
                (d_model=d_model,nhead=n_head,dim_feedforward=dim_feedforward,dropout=dropout)
                for _ in range(n_layers)])
        self.layernorm=nn.LayerNorm(d_model)
        self.inputencoding=InputEncoding()
        self.positionalencoding = PositionalEncoding(d_model=d_model,max_len=max_len)
        self.linear=nn.Linear(d_input, d_model)
    
    def forward(self, x):
        # 卷积
        x=self.inputencoding(x)
        # 位置编码
        PE=self.positionalencoding(x)
        # 线性层到512
        LE=self.linear(x)
        # 相加
        unlayernorm=PE+LE
        # 进行第一次layernorm
        inencoder=self.layernorm(unlayernorm)
        output=inencoder
        # 使用torch的编码器层
        for en_layer in self.encoderlayers:
            output = en_layer(output)
        return output