import torch
from graphml.utils import scatter_split


def test_scatter_split_should_return_a_list_of_tensors_according_to_indexes():
    src = torch.tensor(
        [[2, 1], [5, 6], [3, 0], [-3, 2], [9, 4]], dtype=torch.float)
    indexes = torch.tensor([1, 0, 1, -1, 0])

    result = scatter_split(src, indexes)

    assert len(result) == 3
    assert result[0].tolist() == [[-3, 2]]
    assert result[1].tolist() == [[5, 6], [9, 4]]
    assert result[2].tolist() == [[2, 1], [3, 0]]
