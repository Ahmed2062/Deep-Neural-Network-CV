import numpy as np
import pytest

from compression import huffman


@pytest.mark.parametrize("data", [
    np.random.default_rng(0).geometric(0.3, 5000),   # skewed
    np.random.default_rng(1).integers(0, 32, 5000),  # near-uniform
    np.full(100, 7),                                 # one distinct symbol
    np.array([3, 3, 3, 9]),
])
def test_round_trip(data):
    packed, n_bits, table = huffman.encode(data)
    assert np.array_equal(huffman.decode(packed, n_bits, table, len(data)), data)


def test_prefix_free_and_optimal_for_powers_of_two():
    # Frequencies 1/2, 1/4, 1/8, 1/8 -> optimal lengths 1, 2, 3, 3.
    data = np.array([0] * 4 + [1] * 2 + [2] + [3])
    lengths = huffman.code_lengths(data)
    assert sorted(lengths.values()) == [1, 2, 3, 3]
    codes = huffman.canonical_codes(lengths)
    words = [format(codes[s], f"0{lengths[s]}b") for s in codes]
    assert not any(a != b and b.startswith(a) for a in words for b in words)
