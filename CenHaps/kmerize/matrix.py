"""
Subcommand ``matrix``: build the per-chromosome density matrix.

Input:
  - chrom kmer lib: {chrom_lib}/Chr{XX}.csv (kmer)
  - sample counts:  {sample_count}/Chr{XX}_{samplename}.csv       (kmer, count, density)
Output:
  - density matrix: {output}/Chr{XX}.csv   (kmer, <sample1>, <sample2>, ...)
"""

import os

import polars as pl

from .utils import CHROMS, parse_count_filename


def run_matrix(chrom_lib: str, sample_count: str, output: str) -> None:
    os.makedirs(output, exist_ok=True)

    sample_files = os.listdir(sample_count)

    for chrom in CHROMS:
        kmer_lib = pl.scan_csv(os.path.join(chrom_lib, f"{chrom}.csv"))

        # All count files belonging to the chromosome
        chrom_count_files = [
            f for f in sample_files
            if f.startswith(chrom) and f.endswith(".csv")
        ]

        mat = kmer_lib
        for count_file in chrom_count_files:
            _, samplename = parse_count_filename(count_file)
            sample_density = pl.scan_csv(
                os.path.join(sample_count, count_file)
            ).select(["kmer", "density"])

            mat = (
                mat
                .join(sample_density, on="kmer", how="left")
                .rename({"density": samplename})
                .fill_null(0)
            )

        mat.collect().write_csv(os.path.join(output, f"{chrom}.csv"))
