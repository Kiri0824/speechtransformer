import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IGNORE_ID
from ..PositionalEncoding import PositionalEncoding
from ..GenerateOutputSeq import in_out_process
class Decoder(nn.Module):
    # max_len是输出的最长长度
    def __init__(self, vocab_size=4335,embedding_dim=512,n_layers=6, n_head=8, 
                 d_model=512, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.positionalencoding = PositionalEncoding(d_model=embedding_dim,max_len=max_len)
        self.decoderlayers=nn.ModuleList([nn.TransformerDecoderLayer
                (d_model=d_model,nhead=n_head,dim_feedforward=dim_feedforward,dropout=dropout)
                for _ in range(n_layers)])
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Linear=nn.Linear(max_len,d_model)
    def forward(self, x, encoder_output):
        
        # x: shape (seq_len, batch_size)
        # encoder_output: shape (src_seq_len, batch_size, d_model)
        in_pad,out_pad=in_out_process(x)
        OE=self.embedding(in_pad)
        OPE=self.positionalencoding(in_pad)
        outinput=OE+OPE
        output=outinput
        for de_layer in self.decoderlayers:
            output = de_layer(output,encoder_output)
        
        
        return output