import torch
import torch.nn.functional as F
from torch import nn
from graphml.utils import add_self_edges_to_adjacency_matrix
from torch_scatter import scatter_softmax


class GatLayer(nn.Module):
    def __init__(self, input_feature_dim: torch.Size, output_feature_dim: torch.Size, attention_leakyReLU_slope=0.2, dropout_prob=0.):
        super(GatLayer, self).__init__()

        self._weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float32))
        self._attention_bias_vector = nn.Parameter(
            torch.empty(1, 2*output_feature_dim, dtype=torch.float32))

        self._init_parameters()

        self._dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        self._leaky_relu = nn.LeakyReLU(attention_leakyReLU_slope)

    def _init_parameters(self):
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter)

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor):
        if self._dropout is not None:
                input_matrix = self._dropout(input_matrix)

        weighted_inputs = torch.mm(input_matrix, self._weights_matrix)

        edge_src_idxs, edge_trg_idxs = adjacency_coo_matrix
        nodes = weighted_inputs.index_select(dim=0, index=edge_src_idxs)
        neighbors = weighted_inputs.index_select(dim=0, index=edge_trg_idxs)

        alpha = self._calc_self_attention(nodes, neighbors, edge_src_idxs)

        neighbors = neighbors * alpha
        return torch.zeros_like(weighted_inputs).scatter_add_(
            src=neighbors, index=edge_src_idxs.view(-1, 1).expand_as(neighbors), dim=0)

    def _calc_self_attention(self, nodes: torch.Tensor, neighbors: torch.Tensor, edge_src_idxs: torch.Tensor):
        alpha = torch.mm(
            torch.cat([nodes, neighbors], dim=1), self._attention_bias_vector.t())
        alpha = self._leaky_relu(alpha)

        if self._dropout is not None:
            alpha = self._dropout(alpha)

        return scatter_softmax(alpha, edge_src_idxs, dim=0).view(-1, 1)


class MultiHeadGatLayer(nn.Module):
    def __init__(self, heads_number: int, input_feature_dim: torch.Size, single_head_output_dim: torch.Size, attention_leakyReLU_slope=0.2, dropout_prob=0.,concat=True, activation_function=F.elu):
        super(MultiHeadGatLayer, self).__init__()

        for i in range(heads_number):
            self.add_module("GAT_head_{}".format(i), GatLayer(
                input_feature_dim, single_head_output_dim, attention_leakyReLU_slope,dropout_prob))

        self._concat = concat
        self._activation = activation_function

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor):
        head_outputs = [head(input_matrix, adjacency_coo_matrix)
                        for head in self.children()]
        if self._concat:
            ELU_outputs = [self._activation(output)
                           for output in head_outputs]
            return torch.cat(ELU_outputs, dim=1)
        else:
            mean_output = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
            return self._activation(mean_output)
