from networkx.classes import graph
import torch
from .InternalData import GraphData
from graphml.utils import add_self_edges_to_adjacency_matrix, normalize_matrix, sample_neighbors


class Transform():
    def __call__(self, data_obj: GraphData) -> GraphData:
        d = {}
        for key in data_obj._fields:
            func = getattr(self, f"transform_{key}", None)
            if callable(func):
                d[key] = func(getattr(data_obj, key), data_obj)

        return data_obj._replace(**d)


class NormalizeFeatures(Transform):
    def transform_features_vectors(self, features_vectors: torch.Tensor, graph: GraphData):
        return normalize_matrix(features_vectors)

class AddSelfLoop(Transform):
    def transform_adj_coo_matrix(self, adj_coo_matrix: torch.Tensor, graph: GraphData):
        return add_self_edges_to_adjacency_matrix(adj_coo_matrix, graph.size)

class SubSampleNeighborhoodSize(Transform):
    def __init__(self, sample_size: int):
        self._sample_size = sample_size
        super().__init__()

    def transform_adj_coo_matrix(self, adj_coo_matrix: torch.Tensor, graph: GraphData):
        return sample_neighbors(adj_coo_matrix,self._sample_size)
