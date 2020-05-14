import torch


def add_self_edges_to_adjacency_matrix(adjency_coo_matrix):
    max_node_id = adjency_coo_matrix.max().item()
    number_of_nodes = max_node_id + 1

    self_edges = torch.arange(number_of_nodes).repeat(2, 1)

    return torch.cat([adjency_coo_matrix, self_edges], dim=1)
