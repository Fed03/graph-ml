import torch


def add_self_edges_to_adjacency_matrix(adjency_coo_matrix):
    max_node_id = adjency_coo_matrix.max().item()
    self_edges = torch.arange(
        max_node_id + 1, dtype=adjency_coo_matrix.dtype).repeat(2, 1)

    src_idxs, trg_idxs = adjency_coo_matrix
    mask = src_idxs != trg_idxs

    return torch.cat([adjency_coo_matrix[:, mask], self_edges], dim=1)
