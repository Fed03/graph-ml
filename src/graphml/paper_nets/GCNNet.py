import torch
import torch.nn.functional as F
from typing import Callable, Tuple
from graphml.layers.gcn import GCNLayerFactory


class GCNNet(torch.nn.Module):
    def __init__(self, adjency_coo_matrix: torch.Tensor, input_feature_dim: torch.Size, number_of_classes: int):
        super().__init__()

        layer_factory = GCNLayerFactory(adjency_coo_matrix)
        self._conv1 = layer_factory.create(input_feature_dim, 16)
        self._conv2 = layer_factory.create(
            16, number_of_classes, torch.nn.LogSoftmax(dim=1), .5)

    def forward(self, input_matrix: torch.Tensor):
        x = self._conv1(input_matrix)
        x = self._conv2(x)
        return x

def GCN_model(
    adjency_coo_matrix: torch.Tensor,
    input_feature_dim: torch.Size,
    number_of_classes: int,
    learning_rate=.01,
    weight_decay=5e-4
) -> Tuple[GCNNet, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.optim.Optimizer]:
    net = GCNNet(adjency_coo_matrix, input_feature_dim, number_of_classes)
    return net, F.nll_loss, torch.optim.Adam(net.parameters(), learning_rate,weight_decay=weight_decay)
