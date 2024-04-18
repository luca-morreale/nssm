
from torch.optim.lr_scheduler import *

def create(config, experiment):
    schedulers = []
    ## use sch['opt_idx'] to select which optimizer to apply the scheduler to
    for sch in config:
        schedulers.append(globals()[sch['name']](experiment['optimizers'][sch['opt_idx']], **sch['params']))
    return schedulers
