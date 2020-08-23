import torch
from .InternalData import GraphData
from graphml.utils import add_self_edges_to_adjacency_matrix, normalize_matrix, sample_neighbors, random_positive_pairs


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
        return sample_neighbors(adj_coo_matrix, self._sample_size)


class CalcPositivePairs():
    def __init__(self, walks_num: int, walk_len: int) -> None:
        self._walks_num = walks_num
        self._walk_len = walk_len

    def __call__(self, data_obj: GraphData) -> GraphData:
        print(f"Generating positive pairs for graph {data_obj.name}")
        pairs = random_positive_pairs(
            data_obj.adj_coo_matrix, self._walks_num, self._walk_len)
        return data_obj._replace(positive_pairs=pairs)
