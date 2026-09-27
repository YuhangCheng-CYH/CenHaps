"""Subcommand ``t2t``: whole-genome (T2T) k-mer count generation.

Scan a directory of whole-genome FASTA files (.fasta / .fasta.gz) and
write a per-chromosome k-mer count table for each record found:

    {out_dir}/{record_id}.csv   (columns: kmer, count)

These counts are consumed by the ``centro`` subcommand as the ``--t2t``
input, where they are compared against centromere k-mer counts to
compute centromere specificity scores.
"""

import gzip
import os
from collections import Counter

import polars as pl
from Bio import SeqIO


def get_seq_kmer(seq: str, size: int, step: int = 1) -> list[str]:
    """Slide a window of *size* over *seq* and return every k-mer."""
    return [seq[i:i + size] for i in range(0, len(seq) - size + 1, step)]


def build_sample_kmer_df(kmers: list[str]) -> pl.DataFrame:
    """Build the per-sample k-mer count table.

    Columns: kmer, count
    """
    counter = Counter(kmers)
    return pl.DataFrame({
        "kmer": list(counter.keys()),
        "count": list(counter.values()),
    })


def list_fasta_files(directory: str) -> list[str]:
    """Return sorted .fasta / .fasta.gz filenames inside *directory*."""
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith((".fasta.gz", ".fasta"))
    )


def run_t2t(t2t_dir: str, out_dir: str, k: int = 27) -> None:
    """Generate per-chromosome k-mer counts from whole-genome FASTA files.

    Parameters
    ----------
    t2t_dir : str
        Directory containing whole-genome FASTA files (.fasta / .fasta.gz).
    out_dir : str
        Output directory for per-chromosome k-mer count CSVs.
    k : int
        K-mer size (default: 27).
    """
    os.makedirs(out_dir, exist_ok=True)

    for fasta_file in list_fasta_files(t2t_dir):
        fasta_path = os.path.join(t2t_dir, fasta_file)
        if fasta_file.endswith(".fasta.gz"):
            with gzip.open(fasta_path, "rt") as handle:
                for rec in SeqIO.parse(handle, "fasta"):
                    kmers = get_seq_kmer(str(rec.seq), k)
                    df_kmers = build_sample_kmer_df(kmers)
                    df_kmers.write_csv(os.path.join(out_dir, f"{rec.id}.csv"))
        else:
            with open(fasta_path, "rt") as handle:
                for rec in SeqIO.parse(handle, "fasta"):
                    kmers = get_seq_kmer(str(rec.seq), k)
                    df_kmers = build_sample_kmer_df(kmers)
                    df_kmers.write_csv(os.path.join(out_dir, f"{rec.id}.csv"))
