import torch.nn as nn
import torch
from .PositionalEncoding import PositionalEncoding
from .InputEncoding import InputEncoding
from .out_process import outprocess
from config import device
class Transformer(nn.Module):
    def __init__(self, d_input=159,n_layers=6, n_head=8, embedding_dim=512,
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
    
    def forward(self,input,output,tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        input=self.inputencoding(input)
        PE=self.positionalencoding(input)
        LE=self.inputlinear(input)
        input=PE+LE
        
        out=outprocess(output)
        OE=self.embedding(out)
        OPE=self.positionalencoding(out)
        outinput=OE+OPE
        
        input=input.permute(1, 0, 2)
        outinput=outinput.permute(1, 0, 2)
        
        transformer_out=self.transformer(input,outinput,tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        # Out size = (sequence length, batch_size, num_tokens)
        out=self.outputlinear(transformer_out)
        # (batch_size, sequence length, num_tokens)
        out = out.permute(1, 0, 2)
        out = nn.functional.softmax(out, dim=2)
        max_values, max_indices = torch.max(out, dim=2)
        return max_indices.float()
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
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