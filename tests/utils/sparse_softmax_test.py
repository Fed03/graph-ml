import torch
import torch.nn.functional as F
from functools import reduce
from functional_pipeline import pipeline, tap
from graphml.utils import sparse_softmax
from pytest import approx


def build_expected(input, mask):
    def accumulator(acc, curr):
        idx, el = curr
        if el not in acc:
            acc[el] = list()
        acc[el].append(input[idx])

        return acc

    return pipeline(
        reduce(accumulator, enumerate(mask), dict()),
        [
            lambda x: x.items(),
            (map, lambda x: (x[0], x[1], F.softmax(torch.tensor(x[1])))),
            (map, lambda x: (x[0], {v: x[2][i].item()
                                    for i, v in enumerate(x[1])})),
            dict
        ]
    )


def test_sparse_softmax():
    input = torch.tensor(
        [.45, .2, .9, .90, .20, .90, .8, .8], dtype=torch.float)
    mask = torch.tensor([2, 3, 5, 2, 0, 0, 3, 1])

    result = sparse_softmax(input, mask).tolist()

    input = input.tolist()
    mask = mask.tolist()

    expected_by_group_and_value = build_expected(input, mask)

    for idx, group, value in zip(range(len(input)), mask, input):
        assert result[idx] == approx(expected_by_group_and_value[group][value])
