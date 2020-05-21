import torch
from graphml.utils import scatter_mean


def test_scatter_mean():
    src = torch.tensor([[2,3,2],[5,2,4],[6,3,4]], dtype=torch.float)
    idx = torch.tensor([1,0,1])

    result = scatter_mean(src,idx)

    assert result.size() == (2,3)
    assert result[0].tolist() == [5,2,4]
    assert result[1].tolist() == [4,3,3]
