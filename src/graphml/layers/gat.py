from torch import nn
import torch
from graphml.utils import add_self_edges_to_adjacency_matrix, sparse_softmax


class GatLayer(nn.Module):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, attention_leakyReLU_slope=0.2):
        super(GatLayer, self).__init__()

        self.weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float32))
        self.attention_bias_vector = nn.Parameter(
            torch.empty(1, 2*output_feature_dim, dtype=torch.float32))

        self.__init_parameters()

        self.leaky_relu = nn.LeakyReLU(attention_leakyReLU_slope)

    def __init_parameters(self):
        for parameter in self.parameters():
            # TODO: add gain?
            nn.init.xavier_uniform_(parameter)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor):
        # TODO: move in outer net
        adj = add_self_edges_to_adjacency_matrix(adjacency_coo_matrix)

        self.weighted_inputs = torch.mm(input_matrix, self.weights_matrix)

        edge_src_idxs = adj[0]
        nodes = self.weighted_inputs.index_select(dim=0, index=edge_src_idxs)
        neighbors = self.weighted_inputs.index_select(dim=0, index=adj[1])

        alpha = self.__calc_self_attention(nodes, neighbors, edge_src_idxs)

        neighbors = neighbors * alpha
        return torch.zeros_like(self.weighted_inputs).scatter_add_(
            src=neighbors, index=edge_src_idxs.view(-1,1).expand_as(neighbors), dim=0)

    def __calc_self_attention(self, nodes: torch.Tensor, neighbors: torch.Tensor, edge_src_idxs: torch.Tensor):
        alpha = torch.mm(
            torch.cat([nodes, neighbors], dim=1), self.attention_bias_vector.t())
        alpha = self.leaky_relu(alpha)

        return sparse_softmax(alpha, edge_src_idxs).view(-1, 1)
