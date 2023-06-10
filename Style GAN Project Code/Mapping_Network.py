import torch
from torch import nn

# from Ada_IN_and_InsetNoise import Ada_IN,Inject_Noise


class Weighted_scale_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(Weighted_scale_layer,self).__init__()
        
        self.Inputsize = in_features
        self.Outputsize = out_features
        self.linear = nn.Linear(self.Inputsize, self.Outputsize)
        self.bias   = self.linear.bias
        self.linear.bias = None
        
        nn.init.normal_(self.linear.weight) 
        nn.init.zeros_(self.bias)

    def forward(self,x):
        scale = ( 2/self.Inputsize)**(1/2)
        scaled_input = x * scale
        out = self.linear(scaled_input)  + self.bias
    
        return out

class Pixen_Norm(nn.Module):
    def __init__(self):
        super(Pixen_Norm, self).__init__()
    def forward(self,x):
        sample_mean = torch.mean(x**2, dim=1, keepdim=True)
        ep = 1e-8
        mean_ep = sample_mean+ep
        norm = torch.sqrt(mean_ep)
        return x / norm
    
    
class Mapping_Network(nn.Module):
    def __init__(self, z_dim, w_dim):
        super(Mapping_Network,self).__init__()
        
        self.relu = nn.ReLU()
        
        self.mapping = nn.Sequential(
            Pixen_Norm(),
            Weighted_scale_layer(z_dim, w_dim),self.relu,
            Weighted_scale_layer(w_dim, w_dim),self.relu,
            Weighted_scale_layer(w_dim, w_dim),self.relu,
            Weighted_scale_layer(w_dim, w_dim),self.relu,
            Weighted_scale_layer(w_dim, w_dim),self.relu,
            Weighted_scale_layer(w_dim, w_dim),self.relu,
            Weighted_scale_layer(w_dim, w_dim),self.relu,
            Weighted_scale_layer(w_dim, w_dim),
        )
    
    def forward(self,x):
        return self.mapping(x)
    
