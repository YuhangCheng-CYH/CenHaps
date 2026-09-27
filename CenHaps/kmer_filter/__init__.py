"""kmer_filter: centromere-specific and chromosome-specific k-mer filtering.

A CLI tool for filtering k-mers by specificity:
  - t2t:    whole-genome (T2T) k-mer count generation
  - centro: centromere-specific k-mer filtering
  - chrom:  chromosome-specific k-mer filtering
  - filter: merge centro + chrom results into final specific k-mer set

Run: ``python -m kmer_filter <subcommand>``
"""

__version__ = "1.0.0"

# Rice reference chromosomes Chr01..Chr12
CHROMS: list[str] = [f"Chr{i:02d}" for i in range(1, 13)]
