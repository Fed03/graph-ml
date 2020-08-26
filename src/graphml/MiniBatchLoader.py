from __future__ import annotations
from typing import List, NamedTuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from graphml.utils import sample_neighbors

""" 
class SampledAdjacency(NamedTuple):
    sampled_adj: torch.Tensor
    original_sampled_adj: torch.Tensor

    def pin_memory(self):
        self.sampled_adj = self.sampled_adj.pin_memory()
        self.original_sampled_adj = self.original_sampled_adj.pin_memory()
        return self

    def to(self, *args, **kwargs):
        return SampledAdjacency(
            self.sampled_adj.to(*args, **kwargs),
            self.original_sampled_adj.to(*args, **kwargs)
        ) """


def stable_argsort(arr, dim=-1, descending=False):
    arr_np = arr.detach().cpu().numpy()
    if descending:
        indices = np.argsort(-arr_np, axis=dim, kind='stable')
    else:
        indices = np.argsort(arr_np, axis=dim, kind='stable')
    return torch.from_numpy(indices).long().to(arr.device)


def isin(test: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    trg_uq, uq_idx = torch.unique(trg, return_inverse=True)

    tmp = torch.cat([trg_uq, test])
    tmp_order = stable_argsort(tmp)

    tmp_sorted = tmp[tmp_order]
    flag = torch.cat([tmp_sorted[1:] == tmp_sorted[:-1],
                      torch.tensor([False], dtype=torch.bool, device=trg.device)])

    mask = torch.empty_like(tmp, dtype=torch.bool)
    mask[tmp_order] = flag

    return mask[uq_idx]


def map_tensor_to_new_idxs(trg: torch.Tensor, old_idxs: torch.Tensor) -> torch.Tensor:
    assert trg.dim() == 1 or trg.dim() == 2

    vectors = [trg] if trg.dim() == 1 else [v for v in trg]
    new_idxs = torch.arange(len(old_idxs), device=old_idxs.device)

    for vec in vectors:
        order = torch.argsort(vec)
        uq, rev_idx = vec[order].unique_consecutive(return_inverse=True)
        mask = isin(uq, old_idxs)
        vec[order] = (new_idxs[mask])[rev_idx]
    return torch.stack(vectors) if len(vectors) > 1 else vectors[0]


class BatchStep(NamedTuple):
    target_idxs: torch.Tensor
    batch_idxs: torch.Tensor
    sampled_idxs: torch.Tensor
    sampled_adjs: List[torch.Tensor]

    def to(self, *args, **kwargs) -> BatchStep:
        return BatchStep(
            self.target_idxs.to(*args, **kwargs),
            self.batch_idxs.to(*args, **kwargs),
            self.sampled_idxs.to(*args, **kwargs),
            list(map(lambda x: x.to(*args, **kwargs), self.sampled_adjs))
        )


class MiniBatchLoader(DataLoader):
    def __init__(self, adjacency_coo_matrix: torch.Tensor, neighborhood_sizes: List[int], node_mask: torch.Tensor = None, **kwargs):
        self._adj = adjacency_coo_matrix

        if isinstance(neighborhood_sizes, list):
            self._neighborhood_sizes = neighborhood_sizes
        else:
            raise TypeError("neighborhood_sizes is not List[int]")

        if node_mask is None:
            nodes_number = adjacency_coo_matrix.max().item() + 1
            node_mask = torch.arange(nodes_number)
        elif node_mask.dtype == torch.bool:
            node_mask = node_mask.nonzero(as_tuple=False).view(-1)

        super().__init__(dataset=node_mask.tolist(),
                         collate_fn=self._sample_needed_neighbors, **kwargs)

    def _sample_needed_neighbors(self, node_idxs_batch: List[int]):
        trg_idx = torch.tensor(node_idxs_batch, device=self._adj.device)

        batch_idxs = trg_idx.clone()
        adjs = []
        for step_size in reversed(self._neighborhood_sizes):
            step_adj = self._select_adj_by_src_idxs(batch_idxs)
            if step_size != -1:
                step_adj = sample_neighbors(step_adj, step_size)
            adjs.append(step_adj)
            batch_idxs = torch.unique(step_adj, sorted=True)

        adjs = [map_tensor_to_new_idxs(adj, batch_idxs) for adj in adjs]

        return BatchStep(
            trg_idx,
            batch_idxs,
            map_tensor_to_new_idxs(trg_idx.clone(), batch_idxs),
            reversed(adjs)
        )

    def _select_adj_by_src_idxs(self, src_idxs: List[int]) -> torch.Tensor:
        mask = isin(src_idxs, self._adj[0])
        return self._adj[:, mask]
