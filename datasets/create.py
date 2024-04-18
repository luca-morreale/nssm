
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from .model import ModelDataset
from .surface_map import SurfaceMapDataset
from .seamless_map import SeamlessMapDataset

def create(config, experiment):
    ### Create a dataset and dataloader for each element in the configuration

    dataset = globals()[config['name']](config)

    sampler = None

    kwargs = {
        'sampler':sampler,
        'num_workers':config['num_workers'],
        'pin_memory':config['pin_memory'],
        'shuffle':config['shuffle'] if sampler is None else False,
        'worker_init_fn':lambda id: np.random.seed(torch.initial_seed() // 2**32 + id)
    }
    loader_class = DataLoader

    return loader_class(dataset, config['batch_size'], **kwargs)
