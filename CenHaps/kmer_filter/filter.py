"""Subcommand ``filter``: merge centromere + chromosome specificity into final k-mer set.

Intersect centromere-specific k-mers (from centro) with chromosome-specific
k-mers (from chrom) to produce the final per-chromosome specific k-mer set.

Input:
  - inter-chrom MSB:   {msb}                        (kmer, n_chrom, msb)
  - centro score dir:  {centro_score}/Chr{XX}.csv   (kmer, get_samples, passed_samples, ratio)
Output:
  - specific kmers:    {output}/Chr{XX}.csv         (kmer)
"""

import os

import polars as pl

from . import CHROMS


def run_filter(msb: str, centro_score: str, output: str,
               msb_ratio: float = 0.05) -> None:
    os.makedirs(output, exist_ok=True)

    # Chromosome-specific k-mers:
    #   n_chrom == 1 -> keep all (chromosome-unique)
    #   n_chrom != 1 -> sort by msb descending, keep top msb_ratio portion
    #   msb_ratio == 0 -> only keep n_chrom == 1 kmers
    chrom_speci = pl.read_csv(msb)
    n_multi = chrom_speci.filter(pl.col("n_chrom") != 1)
    filter_nrow = round(n_multi.height * msb_ratio)

    chrom_speci_kmers = (
        chrom_speci.filter(pl.col("n_chrom") == 1)["kmer"].to_list()
        + n_multi.sort("msb", descending=True).head(filter_nrow)["kmer"].to_list()
    )

    # Per-chromosome: keep centromere-specific kmers that are also
    # chromosome-specific (passed_samples == get_samples)
    for chrom in CHROMS:
        cen_speci = pl.read_csv(os.path.join(centro_score, f"{chrom}.csv"))
        cen_speci = cen_speci.filter(
            pl.col("passed_samples") == pl.col("get_samples")
        )

        speci_kmers = cen_speci.select(pl.col("kmer")).filter(
            pl.col("kmer").is_in(chrom_speci_kmers)
        )
        speci_kmers.write_csv(os.path.join(output, f"{chrom}.csv"))
