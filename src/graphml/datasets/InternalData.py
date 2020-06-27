import torch
from typing import NamedTuple


class InternalData(NamedTuple):
    features_vectors: torch.Tensor
    labels: torch.Tensor
    adj_coo_matrix: torch.Tensor
    train_mask: torch.Tensor
    test_mask: torch.Tensor
    validation_mask: torch.Tensor
    # TODO: add to() method
