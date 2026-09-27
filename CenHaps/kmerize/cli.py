"""
kmerize CLI: Statistical analysis of k-mers from the hor array on the centromere.

Note: the extract step (bed + fasta -> cen155) is a standalone shell script
``extract_cen155.sh`` that calls seqkit directly; it is not part of this CLI.
"""

import argparse

from . import kmer, matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kmerize",
        description="Raw centromere data processing: k-mer counts + density matrix.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- kmer -------------------------------------------------------------- #
    p = sub.add_parser("kmer", help="Generate sample kmer counts + chromsome k-mer libs.")
    p.add_argument("--k", type=int, required=True, help="k-mer size")
    p.add_argument("--hor", required=True,
                   help="Dir of HOR array file, named like Chr{XX}_{samplename}.fasta(.gz)")
    p.add_argument("--o-sample", required=True,
                   help="Output dir of sample kmer counts: named like Chr{XX}_{samplename}.csv")
    p.add_argument("--o-chrom-lib", required=True,
                   help="Output dir of chromsome k-mer libs: named like Chr{XX}.csv")

    # -- matrix ------------------------------------------------------------ #
    p = sub.add_parser("matrix", help="Build kmer density matrix.")
    p.add_argument("--chrom-lib", required=True,
                   help="Dir of chromsome k-mer libs: named like Chr{XX}.csv")
    p.add_argument("--sample-count", required=True,
                   help="Dir dir of sample kmer counts: named like Chr{XX}_{samplename}.csv")
    p.add_argument("--output", required=True,
                   help="Output dir of kmer density matrix: Chr{XX}.csv")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "kmer":
        kmer.run_kmer(
            hor=args.hor,
            k=args.k,
            o_sample=args.o_sample,
            o_chrom_lib=args.o_chrom_lib,
        )
    elif args.command == "matrix":
        matrix.run_matrix(
            chrom_lib=args.chrom_lib,
            sample_count=args.sample_count,
            output=args.output,
        )


if __name__ == "__main__":
    main()
