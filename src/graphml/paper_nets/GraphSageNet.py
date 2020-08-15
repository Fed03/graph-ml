from __future__ import annotations
from itertools import repeat
from typing import Callable, List, Optional
from graphml.ModelRunner import EpochStat
from graphml.datasets.InternalData import GraphData
from graphml.layers.graph_sage import GraphSAGELayer
import torch
import torch.nn.functional as F

from graphml.layers.sage_aggregators import MeanAggregator

class GraphSageNet(torch.nn.Module):
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int):
        super().__init__()

        self._convs = torch.nn.ModuleList([
            GraphSAGELayer(MeanAggregator(input_feature_dim,256)),
            GraphSAGELayer(MeanAggregator(256,number_of_classes),lambda x: x)
        ])
    
    def forward(self, input_matrix: torch.Tensor, *adjs: torch.Tensor):
        assert len(adjs) == len(self._convs) or len(adjs) == 1
        adjs = adjs if len(adjs) != 1 else repeat(adjs[0], len(self._convs))

        x = input_matrix
        for conv, adj in zip(self._convs, adjs):
            x = conv(x, adj)
        return x


class GraphSageSupervisedModel():
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int, learning_rate=.01):
        self._net = GraphSageNet(input_feature_dim, number_of_classes)
        self._loss_fn = F.cross_entropy
        self._optim = torch.optim.Adam(
            self._net.parameters(), learning_rate)
        # pull up
        self._stop_requested = False

    # pull up
    def to(self, *args, **kwargs) -> GraphSageSupervisedModel:
        self._net.to(*args, **kwargs)
        return self

    def fit(self, epochs: int, train_data: GraphData, validation_data: GraphData, *callbacks: Optional[Callable[[GraphSageSupervisedModel, EpochStat], None]]) -> List[EpochStat]:
        self._train_data = train_data
        self._validation_data = validation_data

        return self._internal_fit(epochs, *callbacks)