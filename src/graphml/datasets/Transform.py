import torch
from .InternalData import InternalData
from graphml.utils import normalize_matrix


class Transform():
    def __call__(self, data_obj: InternalData) -> InternalData:
        d = {}
        for key in data_obj._fields:
            func = getattr(self, f"transform_{key}", None)
            if callable(func):
                d[key] = func(getattr(data_obj, key))

        return data_obj._replace(**d)


class NormalizeFeatures(Transform):
    def transform_features_vectors(self, features_vectors: torch.Tensor):
        return normalize_matrix(features_vectors)
