import os
import torch
import pickle
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Callable
from .InternalData import InternalData
from scipy.sparse.csr import csr_matrix
from graphml.utils import build_adj_matrix_from_dict, make_undirected_adjacency_matrix


class PlanetoidDatasetLoader():
    url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    files = ["allx", "ally", "graph", "test.index", "tx", "ty", "x", "y"]

    def __init__(self, dataset_name: str, base_path: str, transform: Callable[[InternalData], InternalData] = None):
        if dataset_name not in ["pubmed", "cora", "citeseer"]:
            raise ValueError("dataset_name")

        self._dataset_name = dataset_name
        self._root_path = os.path.join(base_path, "data", self._dataset_name)
        self._transform = transform if transform else lambda x: x

    def load(self) -> InternalData:
        self._download_dataset()
        self._process()
        return self._internal_data

    @property
    def _pretty_name(self):
        return self._dataset_name.capitalize()

    @property
    def _raw_folder(self):
        return os.path.join(self._root_path, "raw")

    @property
    def _processed_file_path(self):
        return os.path.join(self._root_path, f"{self._dataset_name}.processed.pt")

    @property
    def _raw_file_names(self) -> List[str]:
        return [self._raw_file_name(file) for file in self.files]

    def _raw_file_name(self, file_name: str) -> str:
        return f"ind.{self._dataset_name}.{file_name}"

    def _download_dataset(self):
        if os.path.exists(self._raw_folder):
            print(f"The {self._pretty_name} dataset is already downloaded.")
        else:
            os.makedirs(self._raw_folder)

            print(f"Downloading {self._pretty_name} dataset files...")
            for file_name in tqdm(self._raw_file_names):

                response = requests.get(f"{self.url}/{file_name}")
                with open(os.path.join(self._raw_folder, file_name), "wb") as target:
                    target.write(response.content)
            print("Download completed.")

    def _process(self):
        if not os.path.exists(self._processed_file_path):
            data = self._process_raw_files()
            torch.save(data, self._processed_file_path)
        else:
            data = torch.load(self._processed_file_path)

        self._internal_data = self._transform(data)

        print(f"{self._pretty_name} dataset correctly loaded.")

    def _process_raw_files(self) -> InternalData:
        datas = {file_name: self._load_binary_file(
            file_name) for file_name in self.files if file_name != "test.index"}
        datas["test.index"] = self._load_text_file("test.index")

        features_vectors = torch.cat(
            [datas["allx"], torch.empty_like(datas["tx"])], dim=0)
        features_vectors[datas["test.index"]] = datas["tx"]

        labels = torch.cat(
            [datas["ally"], torch.zeros_like(datas["ty"])], dim=0)
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

        return InternalData(self._pretty_name, features_vectors, labels, make_undirected_adjacency_matrix(datas["graph"]), train_mask, test_mask, val_mask)

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
