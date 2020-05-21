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


def make_undirected_adjacency_matrix(adjency_coo_matrix: torch.Tensor):
    src_idxs, trg_idxs = adjency_coo_matrix

    new_src_idxs = torch.cat([src_idxs, trg_idxs], dim=0)
    new_trg_idxs = torch.cat([trg_idxs, src_idxs], dim=0)

    return torch.stack([new_src_idxs, new_trg_idxs], dim=0).unique(dim=1)


def scatter_mean(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    N = index.max().item() + 1

    sum_size = list(src.size())
    sum_size[0] = N

    _, broadcasted_index = torch.broadcast_tensors(src, index.view(-1, 1))
    sum = torch.zeros(sum_size, dtype=src.dtype, device=src.device).scatter_add_(
        src=src, index=broadcasted_index, dim=0)

    denominator = torch.zeros(N,1, dtype=src.dtype, device=src.device).scatter_add_(
        src=torch.ones(index.size(0),1, dtype=src.dtype, device=src.device), index=index.view(-1, 1), dim=0
    )

    return sum / denominator
