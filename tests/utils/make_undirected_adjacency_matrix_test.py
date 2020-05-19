import torch
from graphml.utils import make_undirected_adjacency_matrix


def test_make_undirected_adjacency_matrix():
    adj = torch.tensor([[1, 5, 4, 2], [0, 8, 3, 4]])

    result = make_undirected_adjacency_matrix(adj).t().tolist()

    assert len(result) == 8

    assert [1, 0] in result
    assert [0, 1] in result

    assert [5, 8] in result
    assert [8, 5] in result

    assert [4, 3] in result
    assert [3, 4] in result

    assert [2, 4] in result
    assert [4, 2] in result


def test_give_an_already_undirected_adjacency_then_make_undirected_adjacency_matrix_should_do_nothing():
    adj = torch.tensor([[1, 5, 0, 8, 1], [0, 8, 1, 5, 1]])

    result = make_undirected_adjacency_matrix(adj).t().tolist()

    assert len(result) == 5

    assert [1, 0] in result
    assert [0, 1] in result

    assert [5, 8] in result
    assert [8, 5] in result
    assert [1, 1] in result
