import torch.nn as nn
from models import utils, layers, normalization

default_initializer = layers.default_init

"""
Followed the implemetation in paper
"From data to functa: Your data point is a function
and you can treat it like one"
"""

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

        y = self.GroupNorm_1(y)
        y = nn.SiLU()(y)
        y = nn.Dropout(p=0.2)(y)
        y = self.hidden_layer_1(y)
        y = x+y

        return y

class Residual_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048):
        super().__init__()

        self.hidden_layer_0 = nn.Linear(in_dim, hidden_dim)
        self.block1 = ResMLP_block(hidden_dim)
        self.block2 = ResMLP_block(hidden_dim)
        self.block3 = ResMLP_block(hidden_dim)
        self.block4 = ResMLP_block(hidden_dim)
        self.block5 = ResMLP_block(hidden_dim, time_embedding=False)
        self.block6 = ResMLP_block(hidden_dim)
        self.block7 = ResMLP_block(hidden_dim)
        self.block8 = ResMLP_block(hidden_dim)
        self.block9 = ResMLP_block(hidden_dim)
        self.block10 = ResMLP_block(hidden_dim)
        self.GroupNorm = nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6)
        self.hidden_layer_1 = nn.Linear(hidden_dim, in_dim)
        nn.init.zeros_(self.hidden_layer_1.weight)
        nn.init.zeros_(self.hidden_layer_1.bias)


    def forward(self, x, t):

        x = (x.squeeze(dim=1)).squeeze(dim=-1)
        x = self.hidden_layer_0(x)
        y1 = self.block1(x, t)
        y2 = self.block2(y1, t)
        y3 = self.block3(y2, t)
        y4 = self.block4(y3, t)
        y5 = self.block5(y4, t)
        y6 = self.block6(y5+y4, t)
        y7 = self.block7(y6+y3, t)
        y8 = self.block8(y7+y2, t)
        y9 = self.block9(y8+y1, t)
        y10 = self.block10(y9+x, t)
        y11 = self.GroupNorm(y10)
        y11 = nn.SiLU()(y11)
        y11 = self.hidden_layer_1(y11)

        return (y11.unsqueeze(dim=1)).unsqueeze(dim=-1)

class Residual_MLP_Small(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024):
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

        x = (x.squeeze(dim=1)).squeeze(dim=-1)
        x = self.hidden_layer_0(x)
        y1 = self.block1(x, t)
        y2 = self.block2(y1, t)
        y3 = self.block3(y2, t)
        y4 = self.block4(y3, t)
        y5 = self.block5(y4+y3, t)
        y6 = self.block6(y5+y2, t)
        y7 = self.block7(y6+y1, t)
        y8 = self.block8(y7+x, t)
        y9 = self.GroupNorm(y8)
        y9 = nn.SiLU()(y9)
        y9 = self.hidden_layer_1(y9)

        return (y9.unsqueeze(dim=1)).unsqueeze(dim=-1)