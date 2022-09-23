import torch
import torch.nn as nn
import functools
#import layers
#import utils

from . import utils, layers, normalization

default_initializer = layers.default_init

class ResMLP_block(nn.Module):
    def __init__(self, hidden_dim=2048, time_embedding=True):
        super().__init__()
        self.time_embedding = time_embedding
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6)
        self.hidden_layer_0 = nn.Linear(hidden_dim, hidden_dim)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.hidden_layer_1.weight)
        nn.init.zeros_(self.hidden_layer_1.bias)

        # For time embedding
        self.nf = nf = int(hidden_dim/4)
        self.act = act = nn.SiLU()
        modules = [nn.Linear(nf, nf * 4)]
        modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
        nn.init.zeros_(modules[0].bias)
        modules.append(nn.Linear(nf * 4, nf * 4))
        modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
        nn.init.zeros_(modules[1].bias)
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, t):
        modules = self.all_modules
        temb = layers.get_timestep_embedding(t, self.nf)
        temb = modules[0](temb)
        temb = modules[1](self.act(temb))

        y = self.GroupNorm_0(x)
        y = nn.SiLU()(y)
        y = self.hidden_layer_0(y)
        if self.time_embedding == True:
            y = y+temb
        else:
            y = y

        y = self.GroupNorm_1(y)
        y = nn.SiLU()(y)
        y = nn.Dropout(p=0.2)(y)
        y = self.hidden_layer_1(y)
        y = x+y

        return y

#block1 = ResMLP_block(512, 1024)
#input = torch.randn(227,512)
#t = torch.rand(input.shape[0])
#block1(input, t)

@utils.register_model(name='residual_mlp')
class Residual_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048):
        super().__init__()

        self.hidden_layer_0 = nn.Linear(in_dim, hidden_dim)
        self.block1 = ResMLP_block(hidden_dim)
        self.block2 = ResMLP_block(hidden_dim)
        self.block3 = ResMLP_block(hidden_dim)
        self.block4 = ResMLP_block(hidden_dim, time_embedding=False)
        self.block5 = ResMLP_block(hidden_dim)
        self.block6 = ResMLP_block(hidden_dim)
        self.block7 = ResMLP_block(hidden_dim)
        self.block8 = ResMLP_block(hidden_dim)
        self.GroupNorm = nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6)
        self.hidden_layer_1 = nn.Linear(hidden_dim, in_dim)
        nn.init.zeros_(self.hidden_layer_1.weight)
        nn.init.zeros_(self.hidden_layer_1.bias)


    def forward(self, x, t):

        if (len(x.size())) >=3:
            x = torch.squeeze(x)

        x = self.hidden_layer_0(x)
        y1 = self.block1(x, t)
        y2 = self.block2(y1, t)
        y3 = self.block3(y2, t)
        y4 = self.block4(y3, t)
        y5 = self.block5(y4+y3, t)
        y6 = self.block4(y5+y2, t)
        y7 = self.block4(y6+y1, t)
        y8 = self.block8(y7+x, t)
        y9 = self.GroupNorm(y8)
        y9 = nn.SiLU()(y9)
        y9 = self.hidden_layer_1(y9)

        return (y9.unsqueeze(dim=1)).unsqueeze(dim=-1)

resmlp = Residual_MLP()
input = torch.randn(227,1, 512, 1)
t = torch.rand(input.shape[0])
print(resmlp(input, t).shape)