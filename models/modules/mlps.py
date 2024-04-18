
from torch.nn import functional as F
from torch.nn import Identity
from torch.nn import Module
from torch.nn import Sequential

from .utils import create_sequential_linear_layer


class ResidualMLPBlock(Module):

    def __init__(self, in_features, act_fun, norm_layer, drop_prob, bias, act_params, out_features=None):
        super().__init__()

        layers = [in_features]*3
        if out_features is not None:
            layers[-1] = out_features

        layer = create_sequential_linear_layer(layers, act_fun, norm_layer, drop_prob, bias, last_act=True, act_params=act_params)

        self.shortcut = Identity()
        if in_features != out_features and out_features is not None:
            self.shortcut = create_sequential_linear_layer([in_features, out_features], act_fun, norm_layer, drop_prob, bias, last_act=False)

        self.residual = Sequential(*layer[:-1])
        self.post_act = layer[-1]


    def forward(self, x):
        x   = self.shortcut(x)
        res = self.residual(x)
        out = self.post_act(res + x)
        return out
