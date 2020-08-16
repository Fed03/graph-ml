import os
import json
import torch
import zipfile
import numpy as np
from networkx import DiGraph
from networkx.readwrite import json_graph
from graphml.datasets.MultiGraphDataset import MultiGraphDataset
from graphml.utils import remove_self_loops, make_undirected_adjacency_matrix
from typing import Callable, List, Union
from graphml.datasets.InternalData import GraphData
from graphml.datasets.BaseDatasetLoader import BaseDatasetLoader


class PPIDataset(BaseDatasetLoader):
    splits = ["test", "train", "valid"]

    def __init__(self, base_path: str, *transform: Callable[[GraphData], GraphData]):
        super().__init__("ppi", base_path, *transform)

    @property
    def _url(self) -> str:
        return "https://data.dgl.ai/dataset"

    @property
    def _raw_file_names(self) -> Union[List[str], str]:
        return "ppi.zip"

    def _split_raw_file(self, split: str, raw_name: str) -> str:
        return os.path.join(self._raw_folder, f"{split}_{raw_name}")

    def _process_raw_files(self) -> MultiGraphDataset:
        split_data = {split: self._process_split(
            split) for split in self.splits}
        num_of_classes = split_data["train"][0].number_of_classes
        features_per_node = split_data["train"][0].features_per_node

        return MultiGraphDataset(split_data["train"], split_data["valid"], split_data["test"], features_per_node, num_of_classes)

    def _process_split(self, split: str) -> List[GraphData]:
        with open(self._split_raw_file(split, "graph.json"), 'r') as f:
            graph = DiGraph(json_graph.node_link_graph(json.load(f)))

        x = np.load(self._split_raw_file(split, "feats.npy"))
        x = torch.from_numpy(x).to(torch.float)

        y = np.load(self._split_raw_file(split, "labels.npy"))
        y = torch.from_numpy(y).to(torch.float)

        graphs_idxs = np.load(self._split_raw_file(split, "graph_id.npy"))
        graphs_idxs = torch.from_numpy(graphs_idxs).to(torch.long)

        graphs_data = []
        for i, g_id in enumerate(torch.unique(graphs_idxs)):
            mask = graphs_idxs == g_id
            adj = self._process_subgraph_adj(mask, graph)
            data = GraphData(
                f"{self._pretty_name}_{split}_{i:02d}", x[mask], y[mask], adj)

            graphs_data.append(self._apply_transforms(data))

        return graphs_data

    def _process_subgraph_adj(self, mask: torch.Tensor, graph: DiGraph) -> torch.Tensor:
        idxs = mask.nonzero().view(-1)

        subgraph = graph.subgraph(idxs.tolist())
        adj = torch.tensor(list(subgraph.edges)).t()
        adj = adj - adj.min()
        adj = remove_self_loops(adj)

        return make_undirected_adjacency_matrix(adj)

    def _dowload_file(self, url: str, file_name: str):
        super()._dowload_file(url, file_name)

        raw_file = os.path.join(self._raw_folder, self._raw_file_names)
        with zipfile.ZipFile(raw_file, 'r') as f:
            f.extractall(self._raw_folder)