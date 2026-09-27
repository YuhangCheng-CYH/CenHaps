"""
Subcommand ``kmer``: generate sample k-mer counts + chromosome k-mer libs.

Input:
  - hor dir:  {hor}/Chr{XX}_{samplename}.fasta(.gz)
Output:
  - sample counts:  {o_sample}/Chr{XX}_{samplename}.csv   (kmer, count, density)
  - chrom kmer lib: {o_chrom_lib}/Chr{XX}.csv             (kmer)
"""

import os
from collections import defaultdict

from .utils import (
    build_chrom_kmer_df,
    build_sample_kmer_df,
    parse_hor_filename,
    read_fasta_kmers,
)


def run_kmer(hor: str, k: int, o_sample: str, o_chrom_lib: str) -> None:
    os.makedirs(o_sample, exist_ok=True)
    os.makedirs(o_chrom_lib, exist_ok=True)

    chrom_kmer_dict: dict = defaultdict(list)

    fasta_list = sorted([f for f in os.listdir(hor) if f.endswith((".fasta.gz", ".fasta"))])

    for fasta_file in fasta_list:
        chrom, samplename = parse_hor_filename(fasta_file)
        kmers, seq_length = read_fasta_kmers(os.path.join(hor, fasta_file), k)

        # Per-sample count table (kmer, count, density)
        df_sample = build_sample_kmer_df(kmers, seq_length)
        df_sample.write_csv(os.path.join(o_sample, f"{chrom}_{samplename}.csv"))

        chrom_kmer_dict[chrom].extend(kmers)

    # Chromosome k-mer libraries (unique + sorted)
    for chrom, kmers in chrom_kmer_dict.items():
        df_chrom = build_chrom_kmer_df(kmers)
        df_chrom.write_csv(os.path.join(o_chrom_lib, f"{chrom}.csv"))
