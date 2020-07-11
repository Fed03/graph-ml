from typing import List, Optional, Union, NamedTuple
import torch
from torch import dtype
from torch.utils.data import DataLoader
from graphml.utils import sample_neighbors


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
        )


class MiniBatchLoader(DataLoader):
    def __init__(self, adjacency_coo_matrix: torch.Tensor, node_mask: Optional[torch.Tensor] = None, neighborhood_sizes: Union[int, List[int]] = None, **kwargs):
        self._adj = adjacency_coo_matrix

        if not neighborhood_sizes:
            self._neighborhood_sizes = []
        elif isinstance(neighborhood_sizes, int):
            self._neighborhood_sizes = [neighborhood_sizes]
        elif isinstance(neighborhood_sizes, list):
            self._neighborhood_sizes = neighborhood_sizes
        else:
            raise TypeError("neighborhood_sizes is not Union[int, List[int]]")

        if node_mask is None:
            nodes_number = adjacency_coo_matrix.max().item() + 1
            node_mask = torch.arange(nodes_number)
        elif node_mask.dtype == torch.bool:
            node_mask = node_mask.nonzero().view(-1)

        super().__init__(dataset=node_mask.tolist(),
                         collate_fn=self._sample_needed_neighbors, **kwargs)

    def _sample_needed_neighbors(self, node_idxs_batch: List[int]):
        sampled_adj = self._select_adj_by_src_idxs(node_idxs_batch)

        if self._neighborhood_sizes:
            adjs = [sample_neighbors(sampled_adj, size)
                    for size in self._neighborhood_sizes]
        else:
            adjs = [sampled_adj]

        node_idxs = torch.unique(torch.cat(adjs))
        adjs = [self._map_adjs_to_new_idxs(node_idxs, adj) for adj in adjs]

        return (torch.tensor(node_idxs_batch, device=node_idxs.device), node_idxs, adjs)

    def _select_adj_by_src_idxs(self, src_idxs: List[int]) -> torch.Tensor:
        mask = torch.full_like(
            self._adj[0], False, dtype=torch.bool, device=self._adj.device)
        for src_idx in src_idxs:
            mask |= self._adj[0] == src_idx

        return self._adj.masked_select(mask).view(2, -1)

    def _map_adjs_to_new_idxs(self, node_idxs: torch.Tensor, adj: torch.Tensor) -> SampledAdjacency:
        sampled = adj.clone()
        for new_idx, old_idx in enumerate(node_idxs):
            sampled[sampled == old_idx] = new_idx

        return SampledAdjacency(sampled, adj)
