import torch.nn as nn
import torch.nn.functional as F
class InputEncoding(nn.Module):
    def __init__(self):
        super(InputEncoding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
    def forward(self, x):
        output=self.conv1(x)
        output=F.relu(output)
        output=F.batch_norm(output)
        output=self.conv2(x)
        output=F.relu(output)
        output=F.batch_norm(output)
        return output
