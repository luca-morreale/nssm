
from .flattener import Flattener


def cut_mesh(V, F, cones):

    flattener = Flattener(V, F, cones)

    flattener.flatten()
    return flattener.get_cut_mesh()
