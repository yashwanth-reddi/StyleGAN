
from torch import nn

from Ada_IN_and_InsetNoise import Ada_IN,Inject_Noise



class Weighted_Scale_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        
        super(Weighted_Scale_Conv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.kernel_square = (kernel_size ** 2)
        self.changed = in_channels * self.kernel_square
        
        self.scale = (2 / self.changed) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        self.channels = self.bias.shape[0]
        
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        
        scaled_iamge = self.conv(x * self.scale)
        
        reshaped_bias = self.bias.view(1, self.channels, 1, 1)
        
        return scaled_iamge + reshaped_bias
    
    
    
class Synthesis_Block(nn.Module):
    def __init__(self, in_channel, out_channel, w_dim=512):
        super(Synthesis_Block, self).__init__()
        
        self.inputchannels  = in_channel
        self.outputchannels = out_channel
        
        self.conv1 = Weighted_Scale_Conv2d(self.inputchannels, self.outputchannels )
        self.conv2 = Weighted_Scale_Conv2d(self.outputchannels , self.outputchannels )
        
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.inject_noise1 = Inject_Noise(out_channel)
        self.inject_noise2 = Inject_Noise(out_channel)
        
        self.adain1 = Ada_IN(out_channel, w_dim)
        self.adain2 = Ada_IN(out_channel, w_dim)
        
    def forward(self, x,w):
        
        x = self.conv1(x)
        x = self.inject_noise1(x)
        x = self.leaky1(x)
        x = self.adain1(x, w)
        
        x = self.conv2(x)
        x = self.inject_noise2(x)
        x = self.leaky2(x)
        x = self.adain2(x, w)

        return x
    
    
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.conv1 = Weighted_Scale_Conv2d(in_channels, out_channels)
        self.conv2 = Weighted_Scale_Conv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.leaky(x)
        
        x = self.conv2(x)
        x = self.leaky(x)
        
        return x

