import torch
from abc import ABC, abstractmethod
from graphml.utils import add_self_edges_to_adjacency_matrix, scatter_mean


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        return self.aggregate(input_matrix, adjacency_coo_matrix)


class MeanAggregator(BaseAggregator):
    def aggregate(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        adj = add_self_edges_to_adjacency_matrix(adjacency_coo_matrix)

        edge_src_idxs = adj[0]
        neighbors = input_matrix.index_select(index=adj[1],dim=0)

        return scatter_mean(neighbors,edge_src_idxs)

