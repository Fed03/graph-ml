from __future__ import annotations
import random
import torch
from torch.distributions import Categorical
from typing import List, Dict
from tqdm import tqdm


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    # return matrix / torch.norm(matrix, p=1, dim=1, keepdim=True)
    row_sum = torch.sum(matrix, dim=1).view(-1, 1)
    normalized = matrix / row_sum
    normalized[torch.isnan(normalized)] = 0

    return normalized


def remove_self_loops(adjency_coo_matrix: torch.Tensor) -> torch.Tensor:
    src, trg = adjency_coo_matrix
    mask = src != trg

    return adjency_coo_matrix[:, mask]


def add_self_edges_to_adjacency_matrix(adjency_coo_matrix: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
    max_node_id = num_nodes - 1 if num_nodes else adjency_coo_matrix.max().item()
    self_edges = torch.arange(
        max_node_id + 1, dtype=adjency_coo_matrix.dtype, device=adjency_coo_matrix.device).repeat(2, 1)

    src_idxs, trg_idxs = adjency_coo_matrix
    mask = src_idxs != trg_idxs

    return torch.cat([adjency_coo_matrix[:, mask], self_edges], dim=1)


def make_undirected_adjacency_matrix(adjency_coo_matrix: torch.Tensor) -> torch.Tensor:
    src_idxs, trg_idxs = adjency_coo_matrix

    new_src_idxs = torch.cat([src_idxs, trg_idxs], dim=0)
    new_trg_idxs = torch.cat([trg_idxs, src_idxs], dim=0)

    return torch.stack([new_src_idxs, new_trg_idxs], dim=0).unique(dim=1)


def scatter_split(src: torch.Tensor, indexes: torch.Tensor) -> List[torch.Tensor]:
    sorted_src = src[indexes.argsort()]
    indexes_count = torch.unique(indexes, return_counts=True)[1]

    return torch.split(sorted_src, indexes_count.tolist(), dim=0)


def sample_neighbors(adjency_coo_matrix: torch.Tensor, sample_size: int) -> torch.Tensor:
    node_idxs, neighbor_idxs = adjency_coo_matrix
    groups = scatter_split(neighbor_idxs, node_idxs)

    sampled_groups = []
    for node_id, group in zip(node_idxs.unique(), groups):
        sample_idxs = torch.randint(len(group), (sample_size,))
        sampled_group_neighbors = group[sample_idxs]
        sampled_groups.append(torch.stack([torch.full_like(
            sampled_group_neighbors, node_id), sampled_group_neighbors], dim=0))

    return torch.cat(sampled_groups, dim=1)


def degrees(adjency_coo_matrix: torch.Tensor) -> torch.Tensor:
    src_idxs = adjency_coo_matrix[0]

    ones = torch.ones_like(src_idxs)

    max_node_id = adjency_coo_matrix.max().item()
    return torch.zeros(max_node_id + 1, device=ones.device, dtype=ones.dtype).scatter_add_(src=ones, index=src_idxs, dim=-1)


def build_adj_matrix_from_dict(dictionary: Dict[int, List[int]]) -> torch.Tensor:
    adj_chunks = []
    for src_idx, trg_idxs in dictionary.items():
        trg_row = torch.tensor(trg_idxs, dtype=torch.long)
        src_row = torch.full_like(trg_row, src_idx, dtype=torch.long)

        adj_chunks.append(torch.stack([src_row, trg_row], dim=0))

    adj = torch.cat(adj_chunks, dim=1).unique(dim=1)
    without_self_loop_mask = adj[0] != adj[1]
    return adj[:, without_self_loop_mask]


def random_positive_pairs(adj: torch.Tensor, walks_num: int, walk_len: int) -> torch.Tensor:
    src, trg = adj

    nodes_map = dict(zip(src.unique().tolist(), map(
        lambda x: x.tolist(), scatter_split(trg, src))))

    pairs = {n:[] for n in nodes_map.keys()}
    for node in tqdm(nodes_map.keys()):
        for _1 in range(walks_num):
            curr_node = node
            for _2 in range(walk_len):
                next_node = random.choice(nodes_map[curr_node])
                while next_node == curr_node:
                    next_node = random.choice(nodes_map[curr_node])
                pairs[node].append(next_node)
                curr_node = next_node

    return torch.stack([torch.as_tensor(pos_list,device=adj.device) for pos_list in pairs.values()])

def negative_sample_idxs(sample_size: int, adj: torch.Tensor) -> torch.Tensor:
    prob = degrees(adj) ** 0.75
    nodes_num = len(prob)
    distrib = Categorical(probs=prob)#.expand((nodes_num,))

    return distrib.sample((sample_size,))#.reshape(nodes_num,-1)
