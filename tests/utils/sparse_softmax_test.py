import torch
from graphml.utils import sparse_softmax

def test_sparse_softmax():
    input = torch.tensor([45,2,9,90,20,90,8,8], dtype=torch.float)
    mask = torch.tensor([2,3,5,2,0,0,3,1])

    result = sparse_softmax(input, mask)