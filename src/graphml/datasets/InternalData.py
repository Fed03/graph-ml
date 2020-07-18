import torch
from typing import NamedTuple, Optional


class InternalData(NamedTuple):
    name: str
    features_vectors: torch.Tensor
    labels: torch.Tensor
    adj_coo_matrix: torch.Tensor
    train_mask: Optional[torch.Tensor] = None
    test_mask: Optional[torch.Tensor] = None
    validation_mask: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        return InternalData._make([t.to(*args, **kwargs) if isinstance(t, torch.Tensor) else t for t in self])

    @property
    def size(self) -> int:
        return len(self.features_vectors)

    @property
    def features_per_node(self) -> int:
        return self.features_vectors.size(-1)

    @property
    def number_of_classes(self) -> int:
        return self.labels.max().item() + 1

    @property
    def train_set(self):
        return self.features_vectors[self.train_mask], self.labels[self.train_mask]

    @property
    def test_set(self):
        return self.features_vectors[self.test_mask], self.labels[self.test_mask]

    @property
    def validation_set(self):
        return self.features_vectors[self.validation_mask], self.labels[self.validation_mask]
