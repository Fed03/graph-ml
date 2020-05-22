import torch
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod


class GraphSAGELayer(nn.Module):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size):
        super(GraphSAGELayer, self).__init__()

        self.weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float32))

        self.__init_parameters()

    def __init_parameters(self):
      for parameter in self.parameters():
          # TODO: add gain?
          nn.init.xavier_uniform_(parameter)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor):
        edge_src_idxs = adjacency_coo_matrix[0]
        
        nodes = input_matrix.index_select(dim=0, index=edge_src_idxs)
        neighbors = input_matrix.index_select(dim=0, index=adjacency_coo_matrix[1])