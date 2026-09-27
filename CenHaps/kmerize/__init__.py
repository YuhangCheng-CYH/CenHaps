"""
kmerize: Statistical analysis of k-mers from the hor array on the centromere.

A CLI tool for the first stage of the centromere k-mer pipeline:
  1. extract  : bed + fasta -> HOR array (seqkit)
  2. kmer     : HOR -> per-sample k-mer counts + chromosome k-mer libs
  3. matrix   : build per-chromosome density matrix

Run: ``python -m kmerize <subcommand>``
"""

__version__ = "1.0.0"
