"""
Canonical Huffman coding for integer symbol streams.

Used for the final stage of Deep Compression: entropy-coding the cluster
indices of the surviving weights and the gaps between their positions.

Canonical Huffman means the decoder only needs each symbol's code *length*
(not the codes themselves) to rebuild the codebook, which keeps the stored
table small: one (symbol, length) pair per distinct symbol.
"""

import heapq
from collections import Counter

import numpy as np


def code_lengths(symbols):
    """Return {symbol: code length in bits} for an iterable of ints."""
    freq = Counter(int(s) for s in symbols)
    if not freq:
        return {}
    if len(freq) == 1:
        # A single distinct symbol still needs 1 bit per occurrence.
        return {next(iter(freq)): 1}

    # Heap items: (frequency, tie-breaker, [symbols in this subtree]).
    heap = [(f, i, [s]) for i, (s, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    lengths = dict.fromkeys(freq, 0)
    counter = len(heap)
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        # Merging two subtrees pushes every symbol in them one level deeper.
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (f1 + f2, counter, s1 + s2))
        counter += 1
    return lengths


def canonical_codes(lengths):
    """Assign canonical codes: sort by (length, symbol), count upwards."""
    codes = {}
    code, prev_len = 0, 0
    for sym, length in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= length - prev_len
        codes[sym] = code
        code += 1
        prev_len = length
    return codes


def encode(symbols):
    """
    Huffman-encode a 1-D integer array.

    Returns (packed_bytes, n_bits, table) where table is an (m, 2) int64
    array of (symbol, code_length) rows. That table is all the decoder needs.
    """
    symbols = np.asarray(symbols, dtype=np.int64).ravel()
    if symbols.size == 0:
        return np.zeros(0, np.uint8), 0, np.zeros((0, 2), np.int64)

    lengths = code_lengths(symbols.tolist())
    codes = canonical_codes(lengths)

    # Vectorised bit emission: look up each symbol's code and length, expand
    # every code into a row of max_len bits, keep only its valid bits.
    uniq = np.array(sorted(lengths), dtype=np.int64)
    pos = np.searchsorted(uniq, symbols)
    code_arr = np.array([codes[s] for s in uniq.tolist()], dtype=np.int64)[pos]
    len_arr = np.array([lengths[s] for s in uniq.tolist()], dtype=np.int64)[pos]

    max_len = int(len_arr.max())
    shifts = len_arr[:, None] - 1 - np.arange(max_len)[None, :]
    bits = (code_arr[:, None] >> np.clip(shifts, 0, None)) & 1
    bits = bits[shifts >= 0].astype(np.uint8)  # row-major keeps code order

    table = np.array([[s, lengths[s]] for s in uniq.tolist()], dtype=np.int64)
    return np.packbits(bits), int(bits.size), table


def decode(packed, n_bits, table, n_symbols):
    """Inverse of encode(). Needs the symbol count to know where to stop."""
    if n_symbols == 0:
        return np.zeros(0, np.int64)

    lengths = {int(s): int(l) for s, l in table}
    # Canonical decoding tables, indexed by code length.
    ordered = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    max_len = ordered[-1][1]
    count = [0] * (max_len + 1)
    for _, l in ordered:
        count[l] += 1
    first_code = [0] * (max_len + 2)
    first_index = [0] * (max_len + 2)
    code = index = 0
    for l in range(1, max_len + 1):
        first_code[l] = code
        first_index[l] = index
        code = (code + count[l]) << 1
        index += count[l]
    sorted_syms = [s for s, _ in ordered]

    bits = np.unpackbits(packed)[:n_bits].tolist()
    out = np.empty(n_symbols, dtype=np.int64)
    code = length = k = 0
    for b in bits:
        code = (code << 1) | b
        length += 1
        offset = code - first_code[length]
        if 0 <= offset < count[length]:
            out[k] = sorted_syms[first_index[length] + offset]
            k += 1
            if k == n_symbols:
                break
            code = length = 0
    if k != n_symbols:
        raise ValueError(f"Bitstream ended after {k} of {n_symbols} symbols")
    return out


def table_bits(table, symbol_bits=32, length_bits=8):
    """Storage cost of a code table, assuming fixed-width fields."""
    return int(len(table)) * (symbol_bits + length_bits)
