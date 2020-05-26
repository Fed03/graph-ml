import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from graphml.utils import add_self_edges_to_adjacency_matrix
from torch_scatter import scatter_mean, scatter_max


class Aggregator(nn.Module):
    pass


class MeanAggregator(Aggregator):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size):
        super().__init__()
        self.weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float32))

        # TODO: add gain?
        nn.init.xavier_uniform_(self.weights_matrix)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        adj = add_self_edges_to_adjacency_matrix(adjacency_coo_matrix)

        edge_src_idxs = adj[0]
        neighbors = input_matrix.index_select(index=adj[1], dim=0)

        mean = scatter_mean(neighbors, edge_src_idxs, dim=0)
        return torch.mm(mean, self.weights_matrix)


class MaxPoolAggregator(Aggregator):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, hidden_feature_dim: torch.Size):
        super().__init__()

        self.weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim + hidden_feature_dim, output_feature_dim, dtype=torch.float32))
        # TODO: add gain?
        nn.init.xavier_uniform_(self.weights_matrix)

        self.fc_net = nn.Linear(input_feature_dim, hidden_feature_dim)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        edge_src_idxs, edge_trg_idxs = adjacency_coo_matrix

        neighbors = input_matrix.index_select(index=edge_trg_idxs, dim=0)
        neighbors_repr = F.relu(self.fc_net(neighbors))
        aggregated_neighbors = scatter_max(
            neighbors_repr, edge_src_idxs, dim=0)

        concat = torch.cat([input_matrix, aggregated_neighbors], dim=1)
        return torch.mm(concat, self.weights_matrix)
