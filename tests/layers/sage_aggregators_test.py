import torch
from graphml.layers.sage_aggregators import *
from pytest import approx


def test_MeanAggregator_should_calc_the_mean_between_every_node_features_and_its_neighbors():
    input_matrix = torch.tensor([[2, 3, 5], [3, 8, 1], [4, 1, 1.5], [
                                10, 4, 3], [4, 1, 0]], dtype=torch.float)
    undirected_adjacency_matrix = torch.tensor(
        [[1, 3, 2, 0, 2, 0, 1, 3, 4, 4], [0, 1, 3, 4, 4, 1, 3, 2, 0, 2]])

    aggr = MeanAggregator()
    result = aggr(input_matrix, undirected_adjacency_matrix)

    assert result.size() == input_matrix.size()
    assert result[0].tolist() == [3, 4, 2]
    assert result[1].tolist() == [5, 5, 3]
    assert result[2].tolist() == [6, 2, 1.5]
    assert approx(result[3].tolist(), rel=0.01) == [5.66, 4.33, 1.83]
    assert approx(result[4].tolist(), rel=0.01) == [3.33, 1.66, 2.16]
