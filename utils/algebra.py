
import numpy as np
import torch


def normalize_vector(vector):
    norm = vector.pow(2).sum(dim=-1, keepdims=True)
    return vector / norm

def rotation_2D(theta, return_array=False):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]]).T
    if return_array:
        return R
    return torch.from_numpy(R).float()
