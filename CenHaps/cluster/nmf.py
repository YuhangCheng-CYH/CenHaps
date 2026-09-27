"""Subcommand ``nmf``: iterative subsampling + NMF distance matrix.

Input:
  - density matrix dir: {density_mat}/Chr{XX}.csv   (kmer, <sample>...)
Output:
  - distance matrix:    {output}/Chr{XX}.csv     (same filename, mean euclidean distance)
"""

import math
import os

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import NMF


def _run_single_iteration(mat_full: np.ndarray, sample_fraction: float,
                          k_min: int, k_max: int, seed: int, i: int) -> np.ndarray:
    """Run one NMF iteration and return the pairwise euclidean distance matrix.

    Each iteration gets an independent RNG (seed + i) so that k-mer subsampling,
    k selection, and NMF initialization are all reproducible yet diverse.
    """
    rng = np.random.RandomState(seed + i)

    # Subsample k-mers (row dimension, without replacement)
    n_kmers = mat_full.shape[0]
    n_select = max(1, int(n_kmers * sample_fraction))
    indices = rng.choice(n_kmers, size=n_select, replace=False)

    # Sparse
    mat_t = mat_full[indices].T
    mat_t_sparse = csr_matrix(mat_t)

    # NMF with random k in [k_min, k_max]
    n_components = rng.randint(k_min, k_max + 1)
    model = NMF(
        n_components=n_components,
        init="nndsvdar",
        beta_loss="kullback-leibler",
        solver="mu",
        random_state=seed + i,
        max_iter=1000,
        tol=1e-4,
    )
    w_mat = model.fit_transform(mat_t_sparse)

    # Pairwise euclidean distance between samples
    return squareform(pdist(w_mat, "euclidean"))


def _iter_nmf_single(density_mat: str, output_data: str, n_iters: int,
                     sample_fraction: float = 0.1, seed: int = 422,
                     n_jobs: int = -1) -> None:
    # Read full table once; convert to numpy for efficient cross-process sharing
    df_full = pl.scan_csv(density_mat).collect()
    mat_full = df_full.drop("kmer").to_numpy()
    n_samples = mat_full.shape[1]

    # Pre-compute constant k range (loop-invariant)
    k_min = round(math.sqrt(n_samples))
    k_max = round(3 * math.sqrt(n_samples))

    # Resolve worker count for batch sizing
    n_workers = os.cpu_count() or 1 if n_jobs == -1 else n_jobs
    batch_size = max(1, n_workers * 2)

    # Parallel batches with incremental sum
    # Memory stays O(batch_size * n^2) instead of O(n_iters * n^2)
    dist_sum = np.zeros((n_samples, n_samples))
    for start in range(0, n_iters, batch_size):
        end = min(start + batch_size, n_iters)
        batch = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_iteration)(
                mat_full, sample_fraction, k_min, k_max, seed, i
            )
            for i in range(start, end)
        )
        dist_sum += np.sum(batch, axis=0)

    dist_mean = dist_sum / n_iters
    df_out = pl.DataFrame(dist_mean)
    df_out.write_csv(output_data)


def run_nmf(density_mat: str, output: str, n_iters: int = 600,
            n_jobs: int = -1) -> None:
    os.makedirs(output, exist_ok=True)

    for rawdata in os.listdir(density_mat):
        _iter_nmf_single(
            density_mat=os.path.join(density_mat, rawdata),
            output_data=os.path.join(output, rawdata),
            n_iters=n_iters,
            n_jobs=n_jobs,
        )
