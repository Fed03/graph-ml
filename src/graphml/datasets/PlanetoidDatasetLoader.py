import os
import torch
import pickle
import numpy as np
from .InternalData import GraphData
from scipy.sparse.csr import csr_matrix
from typing import Any, List, Callable, Union
from graphml.datasets.BaseDatasetLoader import BaseDatasetLoader
from graphml.utils import build_adj_matrix_from_dict, make_undirected_adjacency_matrix


class PlanetoidDatasetLoader(BaseDatasetLoader):
    files = ["allx", "ally", "graph", "test.index", "tx", "ty", "x", "y"]

    def __init__(self, dataset_name: str, base_path: str, *transform: Callable[[GraphData], GraphData]):
        if dataset_name not in ["pubmed", "cora", "citeseer"]:
            raise ValueError("dataset_name")

        super().__init__(dataset_name, base_path, *transform)

    @property
    def _url(self) -> str:
        return "https://github.com/kimiyoung/planetoid/raw/master/data"

    @property
    def _raw_file_names(self) -> Union[List[str], str]:
        return [self._raw_file_name(file) for file in self.files]

    def _raw_file_name(self, file_name: str) -> str:
        return f"ind.{self._dataset_name}.{file_name}"

    def _process_raw_files(self) -> Any:
        datas = {file_name: self._load_binary_file(
            file_name) for file_name in self.files if file_name != "test.index"}
        datas["test.index"] = self._load_text_file("test.index")

        test_count = datas["test.index"].max().item(
        ) - datas["test.index"].min().item() + 1

        features_vectors = torch.cat(
            [datas["allx"], torch.empty(test_count, datas["allx"].size(-1), dtype=datas["allx"].dtype, device=datas["allx"].device)], dim=0)
        features_vectors[datas["test.index"]] = datas["tx"]

        labels = torch.cat(
            [datas["ally"], torch.zeros(test_count, datas["ally"].size(-1), dtype=datas["ally"].dtype, device=datas["ally"].device)], dim=0)
        labels[datas["test.index"]] = datas["ty"]
        labels = labels.argmax(dim=1)

        train_size = len(datas['y'])
        train_idxs = torch.arange(train_size, dtype=torch.long)
        val_idxs = torch.arange(train_size, train_size + 500, dtype=torch.long)
        test_idxs = datas["test.index"]

        dataset_size = len(labels)
        train_mask = torch.full((dataset_size,), False, dtype=torch.bool)
        train_mask[train_idxs] = True

        val_mask = torch.full((dataset_size,), False, dtype=torch.bool)
        val_mask[val_idxs] = True

        test_mask = torch.full((dataset_size,), False, dtype=torch.bool)
        test_mask[test_idxs] = True

        data = GraphData(self._pretty_name, features_vectors, labels, make_undirected_adjacency_matrix(
            datas["graph"]), train_mask, test_mask, val_mask)

        return self._apply_transforms(data)

    def _load_binary_file(self, file_name: str) -> torch.Tensor:
        path = os.path.join(self._raw_folder, self._raw_file_name(file_name))
        with open(path, "rb") as file_obj:
            loaded = pickle.load(file_obj, encoding="latin1")

        if isinstance(loaded, csr_matrix):
            return torch.from_numpy(loaded.toarray())
        elif isinstance(loaded, np.ndarray):
            return torch.from_numpy(loaded)
        elif isinstance(loaded, dict):
            return build_adj_matrix_from_dict(loaded)
        else:
            raise ValueError

    def _load_text_file(self, file_name: str) -> torch.Tensor:
        path = os.path.join(self._raw_folder, self._raw_file_name(file_name))
        with open(path, "r") as file_obj:
            lines = filter(None, file_obj.read().split("\n"))

        return torch.tensor([int(num) for num in lines], dtype=torch.long)
