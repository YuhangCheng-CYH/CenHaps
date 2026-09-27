"""Subcommand ``chrom``: chromosome-specific k-mer filtering via inter-chromosome MSB.

Merge per-chromosome density matrices and compute the inter-chromosome mean
square between (MSB) for each k-mer.

Input:
  - density matrix dir: {input_dir}/Chr{XX}.csv  (kmer, <sample1>, <sample2>, ...)
Output:
  - MSB table:          {output}  (single file: kmer, n_chrom, msb)
"""

import os

import numpy as np
import polars as pl

from . import CHROMS


def run_chrom(density: str, output: str) -> None:
    # Step 1: per-chromosome per-kmer mean across samples
    df_list = []
    n_samples = None

    for chrom in CHROMS:
        df = pl.read_csv(os.path.join(density, f"{chrom}.csv"))
        sample_cols = [c for c in df.columns if c != "kmer"]
        if n_samples is None:
            n_samples = len(sample_cols)

        df_mean = df.select(
            pl.col("kmer"),
            pl.mean_horizontal(pl.col(sample_cols)).alias(chrom),
        )
        df_list.append(df_mean)

    # Step 2: full outer join -> merged matrix (kmer, Chr01..Chr12)
    df_merged = df_list[0]
    for dfm in df_list[1:]:
        df_merged = df_merged.join(dfm, on="kmer", how="full", coalesce=True)

    # Step 3: inter-chromosome MSB per kmer
    #   MSB = sum_i n_i * (x_bar_i - x_bar)^2 / (k - 1)
    #   k         = number of chromosomes where kmer is present
    #   n_i       = number of samples (identical across chromosomes)
    #   x_bar_i   = per-chromosome mean
    #   x_bar     = grand mean (mean of present chromosome means)
    mat = df_merged.select(CHROMS).to_numpy()           # (n_kmer, 12)

    k = np.sum(~np.isnan(mat), axis=1)                  # present chromosomes
    grand_mean = np.nanmean(mat, axis=1)                # x_bar
    dev = mat - grand_mean[:, None]                     # x_bar_i - x_bar
    dev = np.where(np.isnan(mat), 0.0, dev)             # NaN -> 0
    ssb = np.sum(dev ** 2, axis=1)                      # sum (x_bar_i - x_bar)^2

    msb = np.full(len(k), np.nan)
    mask = k > 1
    msb[mask] = n_samples * ssb[mask] / (k[mask] - 1)

    df_msb = pl.DataFrame({
        "kmer": df_merged["kmer"],
        "n_chrom": k,
        "msb": msb,
    })
    df_msb.write_csv(output)
