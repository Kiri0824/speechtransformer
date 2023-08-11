from torch import nn
from torch.nn import functional as F
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()
def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)
    
    
# feature->feature*4->feature
class FeedForward(nn.Module):
    def __init__(self,d_model,mult = 4,dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

    
class Convolution(nn.Module):
    def __init__(self,dim,expansion_factor = 2,kernel_size = 32,dropout = 0.1):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size)
        self.Layer = nn.LayerNorm(dim)
        self.conv1d = nn.Conv1d(dim, inner_dim*2, 1)
        self.glu=GLU(dim=1)
        self.conv1d2 = DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding)
        self.bn = nn.BatchNorm1d(inner_dim)
        self.swish = Swish()
        self.conv1d3 = nn.Conv1d(inner_dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.Layer(x)
        # 转换维度
        x=x.permute(0, 2, 1)
        x = self.conv1d(x)
        # 将x的通道数分成两部分
        x = self.glu(x)
        x = self.conv1d2(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.conv1d3(x)
        x = self.drop(x)
        x=x.permute(0, 2, 1)
        return x