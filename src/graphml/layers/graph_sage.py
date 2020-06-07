import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from graphml.layers.sage_aggregators import Aggregator


class GraphSAGELayer(nn.Module):
    def __init__(self, aggregator: Aggregator, activation_function=F.relu):
        super(GraphSAGELayer, self).__init__()

        if not isinstance(aggregator, Aggregator):
            raise TypeError("Unexpected aggregator")
        
        self.__aggregator = aggregator
        self.__activation = activation_function

    def forward(self, input_matrix: torch.Tensor, adjacency_coo_matrix: torch.Tensor):
        output = self.__activation(self.__aggregator(input_matrix,adjacency_coo_matrix))

        return F.normalize(output)
