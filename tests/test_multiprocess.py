"""Chapter 20 -- the job engine."""

import numpy as np
import pandas as pd

from afml.multiprocess import lin_parts, mp_pandas_obj, nested_parts


def test_lin_parts_covers_every_atom_once():
    for n, k in [(10, 3), (100, 7), (5, 9)]:
        parts = lin_parts(n, k)
        assert parts[0] == 0 and parts[-1] == n
        assert (np.diff(parts) >= 0).all()
        assert len(parts) == min(k, n) + 1


def test_nested_parts_give_heavier_molecules_fewer_atoms():
    """Triangular workloads need unequal partitions for equal wall time."""
    parts = np.diff(nested_parts(1000, 5))
    assert parts.sum() == 1000
    assert parts[0] > parts[-1]                     # early molecules do more work each


def _square(molecule, offset):
    return pd.Series(np.asarray(molecule) ** 2 + offset, index=molecule)


def test_mp_pandas_obj_reassembles_in_order():
    idx = pd.Index(range(50))
    out = mp_pandas_obj(_square, ("molecule", idx), num_threads=1, offset=3)
    assert out.index.equals(idx)
    assert np.array_equal(out.to_numpy(), idx.to_numpy() ** 2 + 3)


def test_parallel_matches_sequential_exactly():
    idx = pd.Index(range(200))
    seq = mp_pandas_obj(_square, ("molecule", idx), num_threads=1, offset=1)
    par = mp_pandas_obj(_square, ("molecule", idx), num_threads=2, mp_batches=2, offset=1)
    pd.testing.assert_series_equal(seq, par)
