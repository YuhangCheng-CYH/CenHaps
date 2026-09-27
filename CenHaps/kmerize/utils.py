"""
Shared utilities for the kmerize pipeline.

Filename conventions (MUST NOT change):
  - hor fasta:   Chr{XX}_{samplename}.fasta(.gz)
  - bed:            Chr{XX}_{samplename}.bed
  - sample counts:  Chr{XX}_{samplename}.csv   (columns: kmer, count, density)
  - chrom kmer lib: Chr{XX}.csv                (columns: kmer)
  - density matrix: Chr{XX}.csv                (columns: kmer, <sample>...)
"""

import gzip
import os
from collections import Counter

import numpy as np
import polars as pl
from Bio import SeqIO

# Rice reference chromosomes Chr01..Chr12
CHROMS: list[str] = [f"Chr{i:02d}" for i in range(1, 13)]


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_hor_filename(filename: str) -> tuple[str, str]:
    """Parse ``Chr{XX}_{samplename}.fasta(.gz)`` -> (chrom, samplename)."""
    chrom = filename.split("_")[0]
    samplename = filename.split("_")[1].split(".")[0]
    return chrom, samplename


def parse_count_filename(filename: str) -> tuple[str, str]:
    """Parse ``Chr{XX}_{samplename}.csv`` -> (chrom, samplename)."""
    chrom = filename.split("_")[0]
    samplename = filename.split("_")[1].removesuffix(".csv")
    return chrom, samplename


# --------------------------------------------------------------------------- #
# k-mer extraction
# --------------------------------------------------------------------------- #
def get_seq_kmer(seq: str, size: int, step: int = 1) -> list[str]:
    """Slide a window of *size* over *seq* and return every k-mer."""
    return [seq[i:i + size] for i in range(0, len(seq) - size + 1, step)]


def read_fasta_kmers(fasta_file: str, k: int) -> tuple[list[str], int]:
    """Read a (gzip) fasta and return (all_kmers, total_sequence_length)."""
    kmers: list[str] = []
    seq_length = 0

    if fasta_file.endswith(".gz"):
        with gzip.open(fasta_file, "rt") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                kmers.extend(get_seq_kmer(str(rec.seq), k))
                seq_length += len(rec.seq)
    else:
        with open(fasta_file, "rt") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                kmers.extend(get_seq_kmer(str(rec.seq), k))
                seq_length += len(rec.seq)        

    return kmers, seq_length


# --------------------------------------------------------------------------- #
# DataFrame builders
# --------------------------------------------------------------------------- #
def build_sample_kmer_df(kmers: list[str], seq_length: int) -> pl.DataFrame:
    """Build the per-sample k-mer count table.

    Columns: kmer, count, density
    where ``density = count * 1000 / seq_length``.
    """
    counter = Counter(kmers)
    return pl.DataFrame({
        "kmer": list(counter.keys()),
        "count": list(counter.values()),
    }).with_columns(
        (pl.col("count") * 1000 / seq_length).alias("density")
    )


def build_chrom_kmer_df(kmers: list[str]) -> pl.DataFrame:
    """Build the chromosome k-mer library (unique, sorted via np.unique)."""
    return pl.DataFrame({"kmer": np.unique(kmers).tolist()})
