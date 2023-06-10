
from torch import nn
import torch

from Blocks import Weighted_Scale_Conv2d,Conv_Block

factors = [1,1,1,1/2,1/4,1/8,1/16,1/32,1/64]


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        
        self.input_channels = in_channels
        
        self.prog_blocks  = nn.ModuleList([])
        self.rgb_layers   = nn.ModuleList([])
        
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            
            conv_input_channels = int(self.input_channels * factors[i])
            
            conv_output_channels = int(self.input_channels * factors[i - 1])
            
            self.prog_blocks.append(Conv_Block(conv_input_channels, conv_output_channels))
            
            self.rgb_layers.append(Weighted_Scale_Conv2d(img_channels, conv_input_channels, kernel_size=1, stride=1, padding=0))

            
        self.initial_rgb = Weighted_Scale_Conv2d(img_channels, self.input_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  

        self.final_block = nn.Sequential(
            Weighted_Scale_Conv2d(self.input_channels + 1, self.input_channels, kernel_size=3, padding=1),self.leaky,
            Weighted_Scale_Conv2d(self.input_channels, self.input_channels, kernel_size=4, padding=0, stride=1),self.leaky,
            Weighted_Scale_Conv2d(self.input_channels, 1, kernel_size=1, padding=0, stride=1)  
        )

    def minibatch_std(self, x):
        
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        mean = torch.std(x, dim=0).mean()
        
        return torch.cat([x, mean.repeat(batch_size, 1, height, width)], dim=1)

    
    def forward(self, x, alpha, steps):
        
        current_step = len(self.prog_blocks) - steps
        
        rgb = self.rgb_layers[current_step]
        out = self.leaky(rgb(x))
        
        if steps == 0: 
            out = self.minibatch_std(out)
            size = out.shape[0]
            out = self.final_block(out)
            reshape = out.view(size, -1)
            return reshape
        
        
        rgb = self.rgb_layers[current_step + 1]
        k = self.avg_pool(x)
        downscaled = self.leaky(rgb(k))
        
        prog_out  = self.prog_blocks[current_step]
        out = self.avg_pool(prog_out(out))

        out = alpha * out + (1 - alpha) * downscaled
        
        
        for step in range(current_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        
        size = out.shape[0]
        out = self.final_block(out)
        reshape = out.view(size,-1)
        
        return reshape
    