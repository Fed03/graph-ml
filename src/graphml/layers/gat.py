from torch import nn
import torch
from graphml.utils import add_self_edges_to_adjacency_matrix


class GatLayer(nn.Module):
    def __init__(self, input_feature_dim, output_feature_dim, attention_leakyReLU_slope=0.2):
        super(GatLayer, self).__init__()

        self.weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float32))
        self.attention_bias_vector = nn.Parameter(
            torch.empty(2*output_feature_dim, dtype=torch.float32))

        self.__init_parameters()

        self.leaky_relu = nn.LeakyReLU(attention_leakyReLU_slope)

    def __init_parameters(self):
        for parameter in self.parameters():
            # TODO: add gain?
            nn.init.xavier_uniform_(parameter)

    def forward(self, input_matrix, adjacency_coo_matrix):
        adj = add_self_edges_to_adjacency_matrix(adjacency_coo_matrix)

        self.weighted_inputs = torch.mm(input_matrix, self.weights_matrix)

        edge_src_idxs = adj[0]
        nodes = self.weighted_inputs.index_select(dim=0, index=edge_src_idxs)
        neighbors = self.weighted_inputs.index_select(dim=0, index=adj[1])

        # calc self attention
        alpha = torch.mv(
            torch.cat([nodes, neighbors], dim=1), self.attention_bias_vector)
        alpha = self.leaky_relu(alpha)

        # softmax
        alpha = alpha.exp()
        neighbors_alpha_sum = torch.zeros_like(
            alpha).scatter_add_(src=alpha, index=adj[0], dim=0)
        alpha = alpha / neighbors_alpha_sum

        neighbors = neighbors * alpha.view(-1, 1)

        out = torch.zeros_like(input_matrix).scatter_add_(
            src=neighbors, index=adj[0], dim=0)


def sparse_softmax(input: torch.Tensor, mask_idxs: torch.Tensor):
    N = mask_idxs.max().item() + 1

    out = input.exp()
    denominator = torch.zeros(N, dtype=input.dtype, device=input.device).scatter_add_(
        src=out, index=mask_idxs, dim=0)

    return out / denominator
