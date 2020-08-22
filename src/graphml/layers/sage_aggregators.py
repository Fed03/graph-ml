import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch_scatter import scatter_mean, scatter_max
from graphml.utils import add_self_edges_to_adjacency_matrix, scatter_split


class Aggregator(nn.Module):
    pass


class MeanAggregator(Aggregator):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size):
        super().__init__()
        self._weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float32))

        # TODO: add gain (relu)?
        nn.init.xavier_uniform_(self._weights_matrix)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        adj = add_self_edges_to_adjacency_matrix(adjacency_coo_matrix, len(input_matrix))

        edge_src_idxs = adj[0]
        neighbors = input_matrix.index_select(index=adj[1], dim=0)

        mean = scatter_mean(neighbors, edge_src_idxs, dim=0)
        return torch.mm(mean, self._weights_matrix)


class MaxPoolAggregator(Aggregator):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, hidden_feature_dim: torch.Size):
        super().__init__()

        self._weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim + hidden_feature_dim, output_feature_dim, dtype=torch.float32))
        # TODO: add gain (relu)?
        nn.init.xavier_uniform_(self._weights_matrix)

        self._fc_net = nn.Linear(input_feature_dim, hidden_feature_dim)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        edge_src_idxs, edge_trg_idxs = adjacency_coo_matrix

        neighbors = input_matrix.index_select(index=edge_trg_idxs, dim=0)
        neighbors_repr = F.relu(self._fc_net(neighbors))
        aggregated_neighbors = scatter_max(
            neighbors_repr, edge_src_idxs, dim=0)[0]

        concat = torch.cat([input_matrix, aggregated_neighbors], dim=1)
        return torch.mm(concat, self._weights_matrix)


class LstmAggregator(Aggregator):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, hidden_feature_dim: torch.Size):
        super().__init__()

        self._weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim + hidden_feature_dim, output_feature_dim, dtype=torch.float32))
        # TODO: add gain (relu)?
        nn.init.xavier_uniform_(self._weights_matrix)

        self._lstm = nn.LSTM(input_size=input_feature_dim,
                             hidden_size=hidden_feature_dim)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        edge_src_idxs, edge_trg_idxs = adjacency_coo_matrix

        neighbors = input_matrix.index_select(index=edge_trg_idxs, dim=0)
        aggregated_neighbors = self._aggregate(neighbors, edge_src_idxs)

        concat = torch.cat([input_matrix, aggregated_neighbors], dim=1)
        return torch.mm(concat, self._weights_matrix)

    def _aggregate(self, neighbors_features: torch.Tensor, edge_src_idxs: torch.Tensor) -> torch.Tensor:
        neighbors_features, edge_src_idxs = self._randomly_permutate_neighbors(
            neighbors_features, edge_src_idxs)
        neighbors_groups = scatter_split(neighbors_features, edge_src_idxs)

        outputs, _ = self._lstm(pack_sequence(
            neighbors_groups, enforce_sorted=False))
        outputs, lenghts = pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs.view(-1, outputs.size(-1))

        last_time_step_idxs = torch.arange(lenghts.size(
            0), device=outputs.device) * torch.max(lenghts).item() + (lenghts - 1)
        return outputs[last_time_step_idxs]

    def _randomly_permutate_neighbors(self, neighbors_features: torch.Tensor, edge_src_idxs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = torch.randperm(edge_src_idxs.size(
            0), device=edge_src_idxs.device)
        return neighbors_features[perm], edge_src_idxs[perm]
