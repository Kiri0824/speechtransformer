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
        if mask is not None:
            values, attention = scaled_dot_product_attention(q, k, v, mask[:,:x.size(1), :])
        else:
            values, attention = scaled_dot_product_attention(q, k, v, None)
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)#32,300,512
        # 上图中linear
        out = self.linear_layer(values)
        return out
class EncoderMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads,drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.conv11=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,padding=(3 - 1) // 2)
        self.conv12=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,padding=(3 - 1) // 2)
        self.conv13=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,padding=(3 - 1) // 2)
        self.conv2=nn.Conv1d(in_channels=2*d_model,out_channels=d_model,kernel_size=3,padding=(3 - 1) // 2)
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        # x:32,300,320
        batch_size, time, d_model = x.size()
        q = x.permute(0, 2, 1)
        q = self.conv11(q)
        q = F.relu(q)
        q = self.dropout(q)
        q = q.permute(0, 2, 1)
        
        batch_size, time, d_model = q.size()
        
        q=q.reshape(batch_size, time, self.num_heads, self.head_dim)
        qf=q.permute(0, 2, 1, 3)
        qt=q.permute(0, 2, 3, 1)
        # qf:32,8,300,64
        
        k = x.permute(0, 2, 1)
        k = self.conv12(k)
        k = F.relu(k)
        k = self.dropout(k)
        k = k.permute(0, 2, 1)
        k=k.reshape(batch_size, time, self.num_heads, self.head_dim)
        kf=k.permute(0, 2, 1, 3)
        kt=k.permute(0, 2, 3, 1)
        
        v = x.permute(0, 2, 1)
        v = self.conv13(v)
        v = F.relu(v)
        v = self.dropout(v)
        v = v.permute(0, 2, 1)
        v=v.reshape(batch_size, time, self.num_heads, self.head_dim)
        vf=v.permute(0, 2, 1, 3)
        vt=v.permute(0, 2, 3, 1)
        
        valuesf, attentionf = scaled_dot_product_attention(qf, kf, vf)
        valuesf = valuesf.permute(0, 2, 1, 3).reshape(batch_size, time, self.num_heads * self.head_dim)
        
        valuest, attentiont = scaled_dot_product_attention(qt, kt, vt)
        valuest = valuest.permute(0, 2, 3, 1).reshape(batch_size, time, self.num_heads * self.head_dim)
        
        values=torch.cat((valuesf,valuest),dim=-1)
        # values:32,300,1024
        values=values.permute(0, 2, 1)
        out = self.conv2(values)
        out = F.relu(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)
        return out
# 返回位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model):
        super(PositionalEncoding,self).__init__()
        self.d_model = d_model
    def forward(self,x):
        # even_i是偶数的位置
        even_i = torch.arange(0, self.d_model, 2).float()
        # denominator是分母
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(x.size(1)).reshape(x.size(1), 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        # stacked是将两个矩阵按照第三个维度拼接
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        # PE的维度是max_sequence_length*d_model
        PE = torch.flatten(stacked, start_dim=1, end_dim=2).to(device)
        x = x + PE[:x.size(1), :]
        return x
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
        self.attention=EncoderMultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(drop_prob)
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout2=nn.Dropout(drop_prob)
    def forward(self,x):
        residual_x=x
        x=self.attention(x)
        x=self.dropout1(x)
        x=self.norm1(x+residual_x)
        residual_x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+residual_x)
        return x

# 编码器包括了输入线性变换到d_model维度
# 编码器是先多头再归一化
class Encoder(nn.Module):
    def __init__(self,d_input,d_model,ffn_hidden,num_heads,drop_prob,num_layers):
        super(Encoder,self).__init__()
        self.linear_in = nn.Linear(d_input, d_model)
        self.conv1=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,stride=2)
        self.conv2=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,stride=2)
        self.dropout = nn.Dropout(p=drop_prob)
        self.position_encoder = PositionalEncoding(d_model)
        self.layers=nn.Sequential(*[EncoderLayer(d_model,ffn_hidden,num_heads,drop_prob) for _ in range(num_layers)])
    def forward(self,x):
        x=self.linear_in(x)
        x=x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x=x.permute(0, 2, 1)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x=self.position_encoder(x).to(device)
        return self.layers(x)
# 返回的是加上位置编码的输入
class EmbeddingPosition(nn.Module):
    # language_to_index是一个字典,包含所有字的索引,传入char_list
    def __init__(self, max_sequence_length, d_model, language_to_index,START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    def batch_tokenize(self, batch, start_token, end_token):
        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)
        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(device)
    def forward(self, x,start_token, end_token): 
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        x = self.position_encoder(x).to(device)
        x = self.dropout(x)
        return x
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads,drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        # self.conv12=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,padding=(3 - 1) // 2)
        # self.conv13=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,padding=(3 - 1) // 2)
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size() #30*300*512
        q = self.q_layer(y)#30*300*512  
        q = q.reshape(batch_size, y.size(1), self.num_heads, self.head_dim)#30*500*8*64
        q = q.permute(0, 2, 1, 3)#30*8*300*64
        batch_size, time, d_model = x.size()
        
        kv = self.kv_layer(x)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        # k = x.permute(0, 2, 1)
        # k = self.conv12(k)
        # k = F.relu(k)
        # k = self.dropout(k)
        # k = k.permute(0, 2, 1)
        # k=k.reshape(batch_size, time, self.num_heads, self.head_dim)
        # k=k.permute(0, 2, 1, 3)
        # v = x.permute(0, 2, 1)
        # v = self.conv13(v)
        # v = F.relu(v)
        # v = self.dropout(v)
        # v = v.permute(0, 2, 1)
        # v=v.reshape(batch_size, time, self.num_heads, self.head_dim)
        # v=v.permute(0, 2, 1, 3)
        values, attention = scaled_dot_product_attention(q, k, v, mask) 
        values = values.permute(0, 2, 1, 3).reshape(batch_size, y.size(1), d_model)
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

    def forward(self, x, y, self_attention_mask):
        _y = y.clone()  #只是为了加法
        # 解码器需要mask
        y = self.mask_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=None)
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
        x,y,self_attention_mask =inputs
        for module in self._modules.values():
            y=module(x,y,self_attention_mask)
        return y
class Decoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
        super(Decoder,self).__init__()
        # 词语嵌入
        self.embedding = EmbeddingPosition(max_sequence_length, d_model, language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layers=SequentialDecoder(*[DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob) for _ in range(num_layers)])
    def forward(self,x,y,self_attention_mask,start_token,end_token):
        # x:32*300*512
        # y:32*300*512
        # mask:300*300
        y = self.embedding(y, start_token, end_token)
        y=self.layers(x,y,self_attention_mask)
        return y 
class Transformer(nn.Module):
    def __init__(self,d_model,d_input,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
        super(Transformer,self).__init__()
        self.encoder=Encoder(d_input,d_model,ffn_hidden,num_heads,drop_prob,num_layers)
        self.decoder=Decoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.linear=nn.Linear(d_model,len(language_to_index))
    def forward(self,x,y,de_mask,start_token,end_token):
        # 只有解码器注意力需要mask,de_cross_mask也是不需要传入
        x=self.encoder(x)
        y=self.decoder(x,y,de_mask,start_token,end_token)
        y=self.linear(y)
        return y