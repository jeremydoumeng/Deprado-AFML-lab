"""Multiprocessing engine  --  AFML Chapter 20.

`mp_pandas_obj` splits a pandas index into ``molecules`` and dispatches one
job per molecule, then concatenates the results.  Every heavy loop in this
package (barrier touching, concurrency counting, sample weights, MDA) is
routed through it, exactly as de Prado structures the book's code.

The single-threaded path is kept deliberately identical to the parallel one
so that results are bit-for-bit reproducible regardless of ``num_threads``.
"""

from __future__ import annotations

import sys
import time
import datetime as dt
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["lin_parts", "nested_parts", "mp_pandas_obj", "process_jobs"]


# ----------------------------------------------------------------------------
# Snippet 20.5 -- partitioning an atom list
# ----------------------------------------------------------------------------
def lin_parts(num_atoms: int, num_threads: int) -> np.ndarray:
    """Partition ``num_atoms`` into (almost) equal, contiguous subsets."""
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    return np.ceil(parts).astype(int)


def nested_parts(num_atoms: int, num_threads: int, upper_triang: bool = False) -> np.ndarray:
    """Partition atoms whose per-atom cost grows linearly (triangular loops).

    Used when each molecule performs work proportional to its position, e.g.
    the co-event / uniqueness matrices, so that every core gets equal load.
    """
    parts: list[float] = [0.0]
    num_threads_ = min(num_threads, num_atoms)
    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)
    parts_arr = np.round(parts).astype(int)
    if upper_triang:  # the first rows are the heaviest
        parts_arr = np.cumsum(np.diff(parts_arr)[::-1])
        parts_arr = np.append(np.array([0]), parts_arr)
    return parts_arr


# ----------------------------------------------------------------------------
# Snippets 20.7 / 20.8 / 20.9 -- job scheduling and progress reporting
# ----------------------------------------------------------------------------
def _expand_call(kargs: dict) -> Any:
    """Unpack a job dict and run it (module-level so it is picklable)."""
    func = kargs.pop("func")
    return func(**kargs)


def _report_progress(job_num: int, num_jobs: int, time0: float, task: str) -> None:
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    stamp = str(dt.datetime.fromtimestamp(time.time()))
    out = (
        f"{stamp} {round(msg[0] * 100, 2)}% {task} done after "
        f"{round(msg[1], 2)} minutes. Remaining {round(msg[2], 2)} minutes."
    )
    sys.stderr.write(out + ("\n" if job_num == num_jobs else "\r"))


def process_jobs(
    jobs: Sequence[dict],
    task: str | None = None,
    num_threads: int = 1,
    verbose: bool = False,
) -> list:
    """Run ``jobs`` either sequentially (num_threads == 1) or on a Pool."""
    if task is None:
        task = jobs[0]["func"].__name__

    if num_threads == 1:
        out, time0 = [], time.time()
        for i, job in enumerate(jobs, 1):
            out.append(_expand_call(dict(job)))
            if verbose:
                _report_progress(i, len(jobs), time0, task)
        return out

    pool = Pool(processes=num_threads)
    outputs, out, time0 = pool.imap_unordered(_expand_call, [dict(j) for j in jobs]), [], time.time()
    try:
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            if verbose:
                _report_progress(i, len(jobs), time0, task)
    finally:
        pool.close()
        pool.join()
    return out


# ----------------------------------------------------------------------------
# Snippet 20.7 -- the public entry point
# ----------------------------------------------------------------------------
def mp_pandas_obj(
    func: Callable,
    pd_obj: tuple[str, Sequence],
    num_threads: int = 1,
    mp_batches: int = 1,
    lin_mols: bool = True,
    verbose: bool = False,
    **kargs,
):
    """Parallelise ``func`` over a partition of ``pd_obj[1]``.

    Parameters
    ----------
    func : callable
        Must accept a keyword argument named ``pd_obj[0]`` (the molecule) and
        return a Series or DataFrame indexed by a subset of the molecule.
    pd_obj : (str, sequence)
        Name of the molecule argument, and the full atom list to split.
    num_threads : int
        1 runs in-process (reproducible, debuggable); >1 uses a process Pool.
    """
    if num_threads is None:
        num_threads = cpu_count()
    num_threads = max(1, int(num_threads))

    atoms = pd_obj[1]
    if lin_mols:
        parts = lin_parts(len(atoms), num_threads * mp_batches)
    else:
        parts = nested_parts(len(atoms), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: atoms[parts[i - 1]: parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)

    out = process_jobs(jobs, num_threads=num_threads, verbose=verbose)

    if len(out) == 0:
        return pd.Series(dtype=float)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series(dtype=out[0].dtype)
    else:
        return out
    df0 = pd.concat(out) if len(out) else df0
    return df0.sort_index()
