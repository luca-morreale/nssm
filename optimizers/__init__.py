
from torch.optim import *


def create(config, experiment):
    optimizers = []
    for opt in config:
        optimizers.append(globals()[opt['name']](experiment['model'].parameters(), **opt['params']))

    return optimizers
