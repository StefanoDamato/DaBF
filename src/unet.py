import math
import torch
from torch import nn
import torch.nn.functional as F

from utils import set_device

dvc = set_device()

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim%2 ==0, 'dim has to be even'
        self.dim = dim

        self.inv_freq = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32)*(-math.log(10000)/self.dim)).to(dvc)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, time):
        sinusoid_in = torch.ger(time.view(-1).float(), self.inv_freq)
        time = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        return self.mlp(time)
    

class LabelEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(10, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        
    def forward(self, labels):
        labels = F.one_hot(labels, 10).to(torch.float32)
        return self.mlp(labels)
    

class Embedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.time_emb = TimeEmbedding(dim)
        self.label_emb = LabelEmbedding(dim)

    def forward(self, l, t):
        return self.label_emb(l) + self.time_emb(t)



class ConvNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)

        self.block = nn.Sequential(
            nn.GroupNorm(1, self.in_channels),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, self.out_channels),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(self.in_channels, self.out_channels, 1) if in_channels != out_channels  else nn.Identity()
        self.emb = Embedding(self.out_channels)

    def forward(self, x, l, t):

        emb = self.emb(l, t)
        x_in = self.in_conv(x)

        x_block = self.block(x_in) + emb[:,:,None,None]
        x_res = self.res_conv(x)

        return x_block + x_res
    


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv_list = nn.ModuleList([
            ConvNetBlock(self.in_channels, self.out_channels),
            ConvNetBlock(self.out_channels, self.out_channels)
        ])
        
        self.down = nn.MaxPool2d(2, 2)
        
    def forward(self, x, l, t):
        for block in self.conv_list:
            x = block(x, l, t)
        return self.down(x), x
    

class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up = nn.Upsample(scale_factor=2)

        self.conv_list = nn.ModuleList([
            ConvNetBlock(self.in_channels*2, self.out_channels),
            ConvNetBlock(self.out_channels, self.out_channels)
        ])

    def forward(self, x, l, t, a):
        x = self.up(x)
        x = torch.cat((a, x), axis=1)
        for block in self.conv_list:
            x = block(x, l, t)
        return x

        

class GatedAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.conv = nn.Conv2d(self.channels, self.channels, 2, 2)

        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.channels, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x, g):
        return self.gate(self.conv(x) + g)*x
    

class MultiHeadAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mha = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, x):

        B, S = x.shape[0], x.shape[-1]
        x = torch.swapaxes(x.reshape(B, self.dim, -1), 1, 2)
        x, _ = self.mha(x,x,x)
        return torch.swapaxes(x, 1,2).reshape((B, self.dim, -1, S))
    
        

class UNet(nn.Module):
    
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.input = nn.Sequential(
            nn.InstanceNorm2d(1),
            nn.Conv2d(1, self.dim, 3, padding=1),
            nn.ReLU()
        )

        self.down = nn.ModuleList([
            DownBlock(self.dim, self.dim*2),
            DownBlock(self.dim*2, self.dim*4)
        ])

        self.bottom = ConvNetBlock(dim*4, dim*4)

        self.att = nn.ModuleList([
            GatedAttention(dim*4),
            GatedAttention(dim*2)
        ])

        self.up = nn.ModuleList([
            UpBlock(self.dim*4, self.dim*2),
            UpBlock(self.dim*2, self.dim)
        ])

        self.top = ConvNetBlock(self.dim, self.dim)
        
        self.output = nn.Conv2d(self.dim, 1, 3, padding=1)
    
    def forward(self, x, l, t):

        residuals = []

        x = self.input(x)

        for down_block in self.down:
            x, res = down_block(x, l, t)
            residuals.append(res)

        x = self.bottom(x, l, t)

        for attention, block in zip(self.att, self.up):
            res = residuals.pop()
            a = attention(res, x)
            x = block(x, l, t, a)

        x = self.top(x, l, t)

        x = self.output(x)
        
        return x