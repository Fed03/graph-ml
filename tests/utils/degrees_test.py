import torch
from graphml.utils import degrees


def test_degrees_of_sparse_adj_matrix_should_do_it():
    adj = torch.tensor([[3, 2, 0, 0, 4], 
                        [2, 0, 3, 4, 3]])

    result = degrees(adj)

    assert len(result) == 5
    assert result[0].item() == 2
    assert result[1].item() == 0
    assert result[2].item() == 1
    assert result[3].item() == 1
    assert result[4].item() == 1
