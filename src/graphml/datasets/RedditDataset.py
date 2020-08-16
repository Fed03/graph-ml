import os
import requests
import torch
import zipfile
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from typing import Callable, List, Union
from graphml.datasets.InternalData import GraphData
from graphml.datasets.BaseDatasetLoader import BaseDatasetLoader
from graphml.utils import make_undirected_adjacency_matrix


class RedditDataset(BaseDatasetLoader):
    def __init__(self, base_path: str, *transform: Callable[[GraphData], GraphData]):
        super().__init__("reddit", base_path, *transform)

    @property
    def _url(self) -> str:
        return "https://data.dgl.ai/dataset"

    @property
    def _raw_file_names(self) -> Union[List[str], str]:
        return "reddit.zip"

    def _process_raw_files(self) -> GraphData:
        data = np.load(self._raw_file("data.npz"))
        x = torch.from_numpy(data["feature"]).to(torch.float)
        y = torch.from_numpy(data["label"]).to(torch.long)
        x_to_split = torch.from_numpy(data["node_types"])

        adj = sp.load_npz(self._raw_file("graph.npz"))
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        adj = torch.stack([row, col], dim=0)

        train_mask = x_to_split == 1
        val_mask = x_to_split == 2
        test_mask = x_to_split == 3

        data = GraphData(self._pretty_name, x, y, make_undirected_adjacency_matrix(adj),
                         train_mask, test_mask, val_mask)

        return self._apply_transforms(data)

    def _raw_file(self, raw_name: str) -> str:
        return os.path.join(self._raw_folder, f"{self._dataset_name}_{raw_name}")

    def _dowload_file(self, url: str, file_name: str):
        raw_file = os.path.join(self._raw_folder, file_name)

        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(raw_file, "wb"), "write", total=int(response.headers.get('content-length', 0))) as target:
            for chunk in response.iter_content(chunk_size=4096):
                target.write(chunk)

        print("Unzipping archive...")
        with zipfile.ZipFile(raw_file, "r") as f:
            f.extractall(self._raw_folder)
