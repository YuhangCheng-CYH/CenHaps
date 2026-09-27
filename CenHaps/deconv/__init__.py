"""deconv: centromere haplotype deconvolution via hierarchical clustering + JAX.

A CLI tool for deconvolving NGS k-mer counts into centromere haplotype
compositions using a hierarchical clustering tree and adaptive k-mer
selection (F-value based).

Run: ``python -m deconv <subcommand>``
"""

__version__ = "1.0.0"

# Rice reference chromosomes Chr01..Chr12
CHROMS: list[str] = [f"Chr{i:02d}" for i in range(1, 13)]
