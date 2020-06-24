from graphml.utils import build_adj_matrix_from_dict


def test_given_a_dict_whose_keys_are_src_idxs_and_values_are_lists_of_trg_idxs_then_it_should_build_the_adj_coo_matrix():
    input = {
        3: [4, 5, 2],
        2: [0, 4],
        1: [0],
        0: [3, 4, 7, 3]
    }

    result = build_adj_matrix_from_dict(input)

    assert result.size() == (2, 10)

    as_list = result.t().tolist()
    assert [3, 4] in as_list
    assert [3, 5] in as_list
    assert [3, 2] in as_list
    assert [2, 0] in as_list
    assert [2, 4] in as_list
    assert [1, 0] in as_list
    assert [0, 3] in as_list
    assert [0, 4] in as_list
    assert [0, 7] in as_list
    assert [0, 3] in as_list
