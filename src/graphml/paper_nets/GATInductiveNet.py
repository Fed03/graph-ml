import torch
import torch.nn.functional as F
from itertools import repeat
from typing import Callable, Tuple
from graphml.layers.gat import MultiHeadGatLayer


class GATInductiveNet(torch.nn.Module):
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int):
        super().__init__()

        self._convs = torch.nn.ModuleList([
            MultiHeadGatLayer(4, input_feature_dim, 256),
            MultiHeadGatLayer(4, 1024, 256),
            MultiHeadGatLayer(6, 1024, number_of_classes,
                              activation_function=torch.nn.LogSoftmax(dim=1), concat=False)
        ])

    def forward(self, input_matrix: torch.Tensor, *adjs: torch.Tensor):
        assert len(adjs) == len(self._convs) or len(adjs) == 1

        x = input_matrix
        adjs = adjs if len(adjs) != 1 else repeat(adjs[0], len(self._convs))
        for idx, conv, adj in zip(range(len(adjs)), self._convs, adjs):
            x = conv(x, adj) if idx != 1 else conv(
                x, adj) + x  # skip conn on inter layer
        return x


def GAT_inductive_model(
    input_feature_dim: torch.Size,
    number_of_classes: int,
    learning_rate=.005
) -> Tuple[GATInductiveNet, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.optim.Optimizer]:
    net = GATInductiveNet(input_feature_dim, number_of_classes)
    return net, F.nll_loss, torch.optim.Adam(net.parameters(), learning_rate)
