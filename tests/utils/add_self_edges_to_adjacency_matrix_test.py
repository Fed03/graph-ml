import torch
from graphml.utils import add_self_edges_to_adjacency_matrix


def test_adding_self_edges_to_adjacency_coo_matrix():
    adj = torch.tensor([[0, 3, 4], [2, 2, 0]])

    result = add_self_edges_to_adjacency_matrix(adj)

    assert result.size(-1) == 8
    edge_list = result.t().tolist()
    assert [0, 0] in edge_list
    assert [1, 1] in edge_list
    assert [2, 2] in edge_list
    assert [3, 3] in edge_list
    assert [4, 4] in edge_list


def test_adding_self_edge_to_an_adjacency_coo_matrix_already_containing_some_of_them():
    adj = torch.tensor([[0, 1], [2, 1]])

    result = add_self_edges_to_adjacency_matrix(adj).t().tolist()

    assert len(result) == 4
    assert [0, 0] in result
    assert [1, 1] in result
    assert [2, 2] in result
    assert [0, 2] in result


def test_the_resulting_tensor_should_have_the_same_properties_as_the_input():
    adj = torch.tensor(
        [[0, 1], [2, 1]], dtype=torch.float64, requires_grad=True)

    result = add_self_edges_to_adjacency_matrix(adj)

    assert result.dtype == torch.float64
    assert result.device == adj.device
    assert result.requires_grad == True
