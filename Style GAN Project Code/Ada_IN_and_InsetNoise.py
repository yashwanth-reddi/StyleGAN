
import torch
from torch import nn

from Mapping_Network import Weighted_scale_layer,Pixen_Norm,Mapping_Network


class Ada_IN(nn.Module):
    def __init__(self, channels, w_dim):
        super(Ada_IN,self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale   = Weighted_scale_layer(w_dim, channels)
        self.style_bias    = Weighted_scale_layer(w_dim, channels)

    def forward(self,x,w):
        x = self.instance_norm(x)
        
        style_scale = self.style_scale(w)
        style_scale_reshaped = style_scale.unsqueeze(-1).unsqueeze(-1)
        
        style_bias  = self.style_bias(w)
        style_bias_reshaped = style_bias.unsqueeze(-1).unsqueeze(-1)
        
        return style_scale_reshaped * x + style_bias_reshaped
    
class Inject_Noise(nn.Module):
    def __init__(self, channels):
        super(Inject_Noise,self).__init__()
        self.channels =  channels
        self.weight = nn.Parameter(torch.zeros(1,self.channels,1,1))

    def forward(self, x):
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        noise_shape = (batch_size, 1, height, width)
        noise = torch.randn(noise_shape)
        noise = noise.to(device = x.device)
        
        return x + noise
