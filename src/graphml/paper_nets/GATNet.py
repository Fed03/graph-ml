import torch
import torch.nn.functional as F
from typing import Callable, Tuple
from graphml.layers.gat import MultiHeadGatLayer
from graphml.utils import add_self_edges_to_adjacency_matrix

class GATNet(torch.nn.Module):
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int):
        super().__init__()

        self._conv1 = MultiHeadGatLayer(8,input_feature_dim,8,dropout_prob=0.6)
        self._conv2 = MultiHeadGatLayer(1,64,number_of_classes,activation_function=torch.nn.LogSoftmax(dim=1),dropout_prob=0.6)

    def forward(self,input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor):
        adj = add_self_edges_to_adjacency_matrix(adjacency_coo_matrix)
        x = self._conv1(input_matrix,adj)
        x = self._conv2(x,adj)
        return x

def GAT_model(
    input_feature_dim: torch.Size,
    number_of_classes: int,
    learning_rate=.005,
    weight_decay=5e-4
) -> Tuple[GATNet, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.optim.Optimizer]:
    net = GATNet(input_feature_dim, number_of_classes)
    return net, F.nll_loss, torch.optim.Adam(net.parameters(), learning_rate,weight_decay=weight_decay)