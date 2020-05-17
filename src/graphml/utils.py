import torch


def add_self_edges_to_adjacency_matrix(adjency_coo_matrix: torch.Tensor):
    max_node_id = adjency_coo_matrix.max().item()
    self_edges = torch.arange(
        max_node_id + 1, dtype=adjency_coo_matrix.dtype).repeat(2, 1)

    src_idxs, trg_idxs = adjency_coo_matrix
    mask = src_idxs != trg_idxs

    return torch.cat([adjency_coo_matrix[:, mask], self_edges], dim=1)


def sparse_softmax(input: torch.Tensor, mask_idxs: torch.Tensor):
    N = mask_idxs.max().item() + 1

    out = input.view(input.size(0)).exp()
    denominator = torch.zeros(N, dtype=input.dtype, device=input.device).scatter_add_(
        src=out, index=mask_idxs, dim=0)[mask_idxs]

    return out / denominator
