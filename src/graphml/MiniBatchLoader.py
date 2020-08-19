from time import perf_counter
from typing import List, Optional, Union, NamedTuple
import torch
from torch import dtype
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

def isin(test:torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    trg_uq, uq_idx = torch.unique(trg,return_inverse=True)

    tmp = torch.cat([trg_uq, test])
    tmp_order = torch.argsort(tmp)

    tmp_sorted = tmp[tmp_order]
    flag = torch.cat([tmp_sorted[1:] == tmp_sorted[:-1],torch.tensor([False], dtype=torch.bool, device=trg.device)])

    mask = torch.empty_like(tmp, dtype=torch.bool)
    mask[tmp_order] = flag

    return mask[uq_idx]


class BatchStep(NamedTuple):
    target_idxs: torch.Tensor
    node_idxs: torch.Tensor
    sampled_adj: List[torch.Tensor]

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
        s = perf_counter()
        trg_idx = torch.tensor(node_idxs_batch, device=self._adj.device)
        
        batch_idxs = trg_idx.clone()
        adjs = []
        for step_size in reversed(self._neighborhood_sizes):
            step_adj = self._select_adj_by_src_idxs(batch_idxs)
            step_adj = sample_neighbors(step_adj,step_size)
            adjs.append(step_adj)
            batch_idxs = torch.unique(step_adj)

        adjs = [self._map_adjs_to_new_idxs(batch_idxs, adj) for adj in adjs]

        l= trg_idx,batch_idxs,self._map_adjs_to_new_idxs(batch_idxs, trg_idx.clone()),reversed(adjs)
        print(f"mini {(perf_counter() -s)}")
        return l

    def _select_adj_by_src_idxs(self, src_idxs: List[int]) -> torch.Tensor:
        mask = isin(src_idxs,self._adj[0])

        l = self._adj[:,mask]
        return l

    def _map_adjs_to_new_idxs(self, node_idxs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for new_idx, old_idx in enumerate(node_idxs):
            adj[adj == old_idx] = new_idx

        # return SampledAdjacency(sampled, adj)
        return adj
