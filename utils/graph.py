
import numpy as np
import networkx as nx


def convert_mesh_to_graph(V, F):
    E = np.vstack((F[:, [0, 1]], F[:, [0, 2]], F[:, [1, 2]]))
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)

    d = V[E[:, 0]] - V[E[:, 1]]
    d = np.sqrt(np.sum(d ** 2, axis=1))

    graph = build_graph(E, d)

    return graph

def build_graph(edges, costs):
    graph = nx.Graph()
    for edge, c in zip(edges, costs):
        graph.add_edge(*edge, length=c)
    return graph

def find_shortest_path(source_idx, target_idx, mesh_graph=None, V=[], F=[]):

    if mesh_graph is None:
        mesh_graph = convert_mesh_to_graph(V, F)
    path = nx.shortest_path(mesh_graph, source=source_idx, target=target_idx, weight='length')

    return path
