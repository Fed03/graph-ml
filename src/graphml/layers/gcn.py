from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from graphml.utils import add_self_edges_to_adjacency_matrix, degrees


class GCNLayerFactory():
    def __init__(self, adjency_coo_matrix: torch.Tensor):
        adj = add_self_edges_to_adjacency_matrix(adjency_coo_matrix)

        self.__normalized_adj_values = GCNLayerFactory.normalized_adj_values(
            adj)
        self.__adjency_coo_matrix = adj

    def create(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, activation_function=F.relu, input_dropout=0.) -> GCNLayerFactory.Layer:
        return GCNLayerFactory.Layer(
            input_feature_dim,
            output_feature_dim,
            self.__adjency_coo_matrix,
            self.__normalized_adj_values,
            activation_function,
            input_dropout
        )

    @staticmethod
    def normalized_adj_values(adj: torch.Tensor) -> torch.Tensor:
        degrees_vector = degrees(adj).float()
        degrees_vector = degrees_vector.pow(-0.5)
        degrees_vector[degrees_vector == math.inf] = 0

        adj_row_idxs, adj_col_indexes = adj
        adj_values = torch.ones_like(adj_row_idxs)

        return degrees_vector[adj_row_idxs] * adj_values * degrees_vector[adj_col_indexes]

    class Layer(nn.Module):
        def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, adj_coo_matrix: torch.Tensor, adj_values: torch.Tensor, activation_function=F.relu, input_dropout = 0.):
            super().__init__()

            self.__adj_coo_matrix = adj_coo_matrix
            self.__adj_values = adj_values
            self.__activation = activation_function
            self.__dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None

            self.weights_matrix = nn.Parameter(torch.empty(
                input_feature_dim, output_feature_dim, dtype=torch.float32))
            # TODO: add gain?
            nn.init.xavier_uniform_(self.weights_matrix)

        def forward(self, input_matrix: torch.Tensor):
            if self.__dropout is not None:
                input_matrix = self.__dropout(input_matrix)

            weighted = torch.mm(input_matrix, self.weights_matrix)

            edge_src_idxs, edge_trg_idxs = self.__adj_coo_matrix
            neighbors = weighted.index_select(index=edge_trg_idxs, dim=0)

            aggregated = scatter_add(
                self.__adj_values.view(-1, 1) * neighbors, edge_src_idxs, dim=0)

            return self.__activation(aggregated)
