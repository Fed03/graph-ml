import torch
from graphml.utils import add_self_edges_to_adjacency_matrix


def test_adding_self_edges_to_adjacency_coo_matrix():
    adj = torch.tensor([[0, 3, 4], [2, 2, 0]])


    result = add_self_edges_to_adjacency_matrix(adj)

    assert result.size(-1) == 8

    edge_list = result.t().tolist()
    assert [0,0] in edge_list
    assert [1,1] in edge_list
    assert [2,2] in edge_list
    assert [3,3] in edge_list
    assert [4,4] in edge_list
