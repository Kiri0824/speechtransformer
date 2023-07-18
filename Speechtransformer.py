import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from config import max_sequence_length,NEG_INF,device
# mask传入的是一个矩阵,需要mask的位置是-inf,不需要mask的位置是0
def scaled_dot_product_attention(q, k, v, mask=None):
    # q:32,8,300,64
    # k:32,8,300,64
    # v:32,8,300,64
    # mask:300,300
    d_k=q.shape[-1]
    scaled=torch.matmul(q,k.transpose(-2,-1))/np.sqrt(d_k)#32,8,300,300
    if mask is not None:
        # torch自动补充
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention =  F.softmax(scaled,dim=-1)
    values=torch.matmul(attention,v)
    return values,attention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)#32,300,512
        # 上图中linear
        out = self.linear_layer(values)

        return out
# 返回位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super(PositionalEncoding,self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
    def forward(self):
        # even_i是偶数的位置
        even_i = torch.arange(0, self.d_model, 2).float()
        # denominator是分母
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        # stacked是将两个矩阵按照第三个维度拼接
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        # PE的维度是max_sequence_length*d_model
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)   #512->2048
        self.linear2 = nn.Linear(hidden, d_model)   #2048->512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1=nn.LayerNorm(d_model)
        self.droupout1=nn.Dropout(drop_prob)
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2=nn.LayerNorm(d_model)
        self.droupout2=nn.Dropout(drop_prob)
    def forward(self,x):
        residual_x=x
        x=self.attention(x,mask=None)
        x=self.droupout1(x)
        x=self.norm1(x+residual_x)
        residual_x=x
        x=self.ffn(x)
        x=self.droupout2(x)
        x=self.norm2(x+residual_x)
        return x
# 编码器包括了输入线性变换到d_model维度
# 编码器是先多头再归一化
class Encoder(nn.Module):
    def __init__(self,d_input,d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length):
        super(Encoder,self).__init__()
        self.linear_in = nn.Linear(d_input, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.layers=nn.Sequential(*[EncoderLayer(d_model,ffn_hidden,num_heads,drop_prob) for _ in range(num_layers)])
    def forward(self,x):
        PE=self.position_encoder().to(device)
        x=self.linear_in(x)
        x=x+PE
        return self.layers(x)
# 返回的是加上位置编码的输入
class EmbeddingPosition(nn.Module):
    # language_to_index是一个字典,包含所有字的索引,传入char_list
    def __init__(self, max_sequence_length, d_model, language_to_index ):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x): 
        x = self.embedding(x)
        pos = self.position_encoder().to(device)
        x = self.dropout(x + pos)
        return x
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model)#512->1024
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size() #30*300*512
        kv = self.kv_layer(x)#30*300*1024
        q = self.q_layer(y)#30*300*512
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)#30*500*8*128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)#30*500*8*64
        kv = kv.permute(0, 2, 1, 3)#30*8*300*128
        q = q.permute(0, 2, 1, 3)#30*8*300*64
        k, v = kv.chunk(2, dim=-1)#K:30*8*300*64 V:30*8*300*64
        # 加入编码器时无需mask
        values, attention = scaled_dot_product_attention(q, k, v, mask) 
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.mask_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()  #只是为了加法
        # 解码器需要mask
        y = self.mask_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y
# 解码器的正向传播输入参数是多个,需要重写
class SequentialDecoder(nn.Sequential):
    def forward(self,*inputs):
        x,y,self_attention_mask,cross_attention_mask =inputs
        for module in self:
            y=module(x,y,self_attention_mask,cross_attention_mask)
        return y
class Decoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,language_to_index):
        super(Decoder,self).__init__()
        # 词语嵌入
        self.embedding = EmbeddingPosition(max_sequence_length, d_model, language_to_index)
        self.layers=SequentialDecoder(*[DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob) for _ in range(num_layers)])
    def forward(self,x,y,self_attention_mask,cross_attention_mask=None):
        # x:32*300*512
        # y:32*300*512
        # mask:300*300
        y = self.embedding(y)
        y=self.layers(x,y,self_attention_mask,cross_attention_mask)
        return y 
class Transformer(nn.Module):
    def __init__(self,d_model,d_input,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,language_to_index):
        super(Transformer,self).__init__()
        self.encoder=Encoder(d_input,d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length)
        self.decoder=Decoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,language_to_index)
        self.linear=nn.Linear(d_model,len(language_to_index))
    def forward(self,x,y,de_mask,de_cross_mask=None):
        # 只有解码器注意力需要mask,de_cross_mask也是不需要传入
        x=self.encoder(x)
        y=self.decoder(x,y,de_mask,de_cross_mask)
        y=self.linear(y)
        return y