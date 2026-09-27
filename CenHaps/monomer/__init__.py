"""monomer: circular centromere monomer unit reconstruction.

A CLI tool that reconstructs circular monomer units from deconvolution
results and NGS read support, in two steps:

  1. prep   - deconvolution results + k-mer component matrix
              -> per-sample k-mer counts (pre_kmer_count/)
  2. cycles - k-mer counts + minimap2 PAF read support
              -> circular monomer units (monomer_res/)

Run: ``python -m monomer <subcommand>``
"""

__version__ = "1.0.0"
