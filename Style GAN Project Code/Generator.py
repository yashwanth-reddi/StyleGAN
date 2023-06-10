from torch import nn
import torch

from Mapping_Network import Weighted_scale_layer,Pixen_Norm,Mapping_Network
from Ada_IN_and_InsetNoise import Ada_IN,Inject_Noise
from Blocks import Weighted_Scale_Conv2d,Synthesis_Block
import torch.nn.functional as F


factors = [1,1,1,1/2,1/4,1/8,1/16,1/32,1/64]


class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, in_channels, img_channels=3):
        super().__init__()
        
        self.starting_cte = nn.Parameter(torch.ones(1, in_channels, 4,4))
        
        self.map = Mapping_Network(z_dim, w_dim)
        
        self.initial_noise1 = Inject_Noise(in_channels)
        self.initial_noise2 = Inject_Noise(in_channels)
        
        self.initial_adain1 = Ada_IN(in_channels, w_dim)
        self.initial_adain2 = Ada_IN(in_channels, w_dim)
        
        
        self.initial_conv   = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.initial_rgb    = Weighted_Scale_Conv2d(in_channels, img_channels, kernel_size = 1, stride=1, padding=0)
        self.leaky          = nn.LeakyReLU(0.2, inplace=True)

        self.prog_blocks  = nn.ModuleList([])
        self.rgb_layers    = nn.ModuleList([self.initial_rgb])
                                           
        for i in range(len(factors)-1):
            conv_input_channels  = int(in_channels * factors[i])
            
            conv_output_channels = int(in_channels * factors[i+1])
            
            self.prog_blocks.append(Synthesis_Block(conv_input_channels, conv_output_channels, w_dim))
            
            self.rgb_layers.append(Weighted_Scale_Conv2d(in_channels=conv_output_channels, out_channels=img_channels, kernel_size = 1, stride=1, padding=0))
        

    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        
        x = self.initial_noise1(self.starting_cte)
        
        x = self.initial_adain1(x,w)

        x = self.initial_conv(x)

        x = self.leaky(self.initial_noise2(x))
        
        out = self.initial_adain2(x, w)

        if steps == 0:
            return self.initial_rgb(x)
        
        for step in range(steps):
            upscaled_image = F.interpolate(out, scale_factor=2, mode = 'bilinear')
            
            out      = self.prog_blocks[step](upscaled_image,w)


        final_upscaled = self.rgb_layers[steps-1](upscaled_image)
        final_out      = self.rgb_layers[steps](out)
        
        
        generated = alpha * final_out
        upscaled = (final_upscaled - alpha* final_upscaled)
        
        changed_to_RGB = torch.tanh(generated + upscaled )
        
        return changed_to_RGB
        
