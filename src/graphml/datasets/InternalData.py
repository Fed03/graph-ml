import torch
from typing import NamedTuple


class InternalData(NamedTuple):
    features_vectors: torch.Tensor
    labels: torch.Tensor
    adj_coo_matrix: torch.Tensor
    train_mask: torch.Tensor
    test_mask: torch.Tensor
    validation_mask: torch.Tensor

    def to(self, *args, **kwargs):
        return InternalData._make([t.to(*args,**kwargs) for t in self])
