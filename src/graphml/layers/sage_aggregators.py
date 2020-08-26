import torch
from torch import nn
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch_scatter import scatter_mean, scatter_max
from graphml.utils import add_self_edges_to_adjacency_matrix, scatter_split
from enum import Enum, auto

class ModelSize(Enum):
    SMALL = auto()
    BIG = auto()

class Aggregator(nn.Module):
    pass


class MeanAggregator(Aggregator):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size):
        super().__init__()
        self._weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float))

        # TODO: add gain (relu)?
        nn.init.xavier_uniform_(self._weights_matrix)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        adj = add_self_edges_to_adjacency_matrix(adjacency_coo_matrix, len(input_matrix))

        edge_src_idxs = adj[0]
        neighbors = input_matrix.index_select(index=adj[1], dim=0)

        mean = scatter_mean(neighbors, edge_src_idxs, dim=0, dim_size=input_matrix.size(0))
        return torch.mm(mean, self._weights_matrix)


class MaxPoolAggregator(Aggregator):
    _sizes: Dict[ModelSize, int] = {
        ModelSize.SMALL: 512,
        ModelSize.BIG: 1024
    }

    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, hidden_feature_dim: Optional[int] = None, model_size: Optional[ModelSize] = None):
        super().__init__()

        hidden_feature_dim = hidden_feature_dim if hidden_feature_dim else self._sizes[model_size]

        self._weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim + hidden_feature_dim, output_feature_dim, dtype=torch.float))
        # TODO: add gain (relu)?
        nn.init.xavier_uniform_(self._weights_matrix)

        self._fc_net = nn.Linear(input_feature_dim, hidden_feature_dim)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        edge_src_idxs, edge_trg_idxs = adjacency_coo_matrix

        neighbors = input_matrix.index_select(index=edge_trg_idxs, dim=0)
        neighbors_repr = F.relu(self._fc_net(neighbors))
        aggregated_neighbors = scatter_max(
            neighbors_repr, edge_src_idxs, dim=0, dim_size=input_matrix.size(0))[0]

        concat = torch.cat([input_matrix, aggregated_neighbors], dim=1)
        return torch.mm(concat, self._weights_matrix)


class LstmAggregator(Aggregator):
    _sizes: Dict[ModelSize, int] = {
        ModelSize.SMALL: 128,
        ModelSize.BIG: 256
    }

    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, hidden_feature_dim: Optional[int] = None, model_size: Optional[ModelSize] = None):
        super().__init__()

        hidden_feature_dim = hidden_feature_dim if hidden_feature_dim else self._sizes[model_size]

        self._weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim + hidden_feature_dim, output_feature_dim, dtype=torch.float))
        # TODO: add gain (relu)?
        nn.init.xavier_uniform_(self._weights_matrix)

        self._lstm = nn.LSTM(input_size=input_feature_dim,
                             hidden_size=hidden_feature_dim)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor) -> torch.Tensor:
        edge_src_idxs, edge_trg_idxs = adjacency_coo_matrix
        neighbors = input_matrix.index_select(index=edge_trg_idxs, dim=0)
        
        lstm_output = self._aggregate(neighbors, edge_src_idxs)
        aggregated_neighbors = torch.zeros(input_matrix.size(0), lstm_output.size(1), device=input_matrix.device)
        aggregated_neighbors[edge_src_idxs.unique()] = lstm_output

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

        lenghts = lenghts.to(outputs.device)

        last_time_step_idxs = torch.arange(lenghts.size(
            0), device=outputs.device) * torch.max(lenghts).item() + (lenghts - 1)
        return outputs[last_time_step_idxs]

    def _randomly_permutate_neighbors(self, neighbors_features: torch.Tensor, edge_src_idxs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = torch.randperm(edge_src_idxs.size(
            0), device=edge_src_idxs.device)
        return neighbors_features[perm], edge_src_idxs[perm]
