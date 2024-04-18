
from .mlp import MLP
from .mlp import ResidualMLP
from .neural_map import NeuralMap
from .neural_map import ParametrizationMap
from .seamless_map import SphereSeamlessMap


def create(config, experiment):
    model = globals()[config['name']](config['structure'])

    return model
