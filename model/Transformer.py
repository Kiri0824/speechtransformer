import torch.nn as nn
import torch
from .PositionalEncoding import PositionalEncoding
from .InputEncoding import InputEncoding
from config import device
class Transformer(nn.Module):
    def __init__(self, d_input=320,n_layers=6, n_head=8, embedding_dim=512,
                 d_model=512,  vocab_size=4335,dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.positionalencoding = PositionalEncoding(d_model=d_model,max_len=max_len)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.inputencoding=InputEncoding()
        self.inputlinear=nn.Linear(d_input, d_model)
        self.transformer=nn.Transformer(d_model=d_model,nhead=n_head,
                                        num_decoder_layers=n_layers,num_encoder_layers=n_layers,
                                        dim_feedforward=dim_feedforward,dropout=dropout,norm_first=True,device=device
                                        )
        self.outputlinear=nn.Linear(d_model,vocab_size)
    
    def forward(self,input_data,output,tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # for i in range(32):
        #     sample = input_data[i].unsqueeze(0)  # 获取单个样本，形状为 [1, 2, 28, 28]
        #     sample=self.inputencoding(sample)  # 对单个样本进行卷积操作，形状为 [1, 16, 28, 28]
        #     input_data[i] = sample.squeeze(0)  # 将单个样本的卷积结果保存到输出中
        # input_data=self.inputencoding(input_data)
        PE=self.positionalencoding(input_data)
        LE=self.inputlinear(input_data)
        input_data=PE+LE

        OE=self.embedding(output)
        OPE=self.positionalencoding(output)
        outinput=OE+OPE
        
        input_data=input_data.permute(1, 0, 2)
        outinput=outinput.permute(1, 0, 2)
        
        transformer_out=self.transformer(input_data,outinput,tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        # Out size = (sequence length, batch_size, num_tokens)
        out=self.outputlinear(transformer_out)
        # (batch_size, sequence length, num_tokens)
        out = out.permute(1, 2, 0)
        
        return out
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)