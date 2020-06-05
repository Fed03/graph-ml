import torch
import pytest
from graphml.utils import sample_neighbors


@pytest.fixture
def setup():
    torch.manual_seed(0)
    yield
    torch.seed()


def test_given_an_adj_matrix_it_should_sample_a_fixed_number_of_neighbors_with_replacement(setup):
    adj = torch.tensor([[0, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0],
                        [3, 5, 4, 7, 6, 2, 8, 9, 4, 6, 3, 7, 4, 4, 2]])

    sample_size = 5
    result = sample_neighbors(adj, sample_size)

    for idx in range(3):
        by_idx = result[1][result[0] == idx]
        assert len(by_idx) == sample_size

        candidates = torch.unique(adj[1][adj[0] == idx])
        for sampled_id in by_idx:
            assert sampled_id in candidates
