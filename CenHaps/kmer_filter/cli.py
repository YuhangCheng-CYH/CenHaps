"""kmer_filter CLI: centromere-specific and chromosome-specific k-mer filtering.

Usage:
  python -m kmer_filter t2t     --t2t-dir ... --output ... [--k 27]
  python -m kmer_filter centro  --cen ... --t2t ... --chrom-lib ... --output ... [--score-threshold 0.9]
  python -m kmer_filter chrom   --density ... --output ...
  python -m kmer_filter filter  --msb ... --centro-score ... --output ... [--msb-ratio 0.05]
"""

import argparse

from . import centro, chrom, filter, t2t


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kmer_filter",
        description="Centromere-specific and chromosome-specific k-mer filtering.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- t2t --------------------------------------------------------------- #
    p = sub.add_parser("t2t", help="Generate whole-genome (T2T) k-mer counts.")
    p.add_argument("--t2t-dir", required=True,
                   help="Dir of whole-genome FASTA files (.fasta / .fasta.gz)")
    p.add_argument("--output", required=True,
                   help="Output dir of per-chromosome k-mer counts: {record_id}.csv")
    p.add_argument("--k", type=int, default=27,
                   help="K-mer size (default: 27)")

    # -- centro ------------------------------------------------------------- #
    p = sub.add_parser("centro", help="Centromere-specific k-mer filtering.")
    p.add_argument("--cen", required=True,
                   help="Dir of centromere k-mer counts named like Chr{XX}_{samplename}.csv")
    p.add_argument("--t2t", required=True,
                   help="Dir of whole-genome k-mer counts named like Chr{XX}_{samplename}.csv")
    p.add_argument("--chrom-lib", required=True,
                   help="Dir of chromosome k-mer libraries named like Chr{XX}.csv")
    p.add_argument("--output", required=True,
                   help="Output dir of  specificity score: Chr{XX}.csv")
    p.add_argument("--score-threshold", type=float, default=0.9,
                   help="Centromere specificity score threshold (default: 0.9)")

    # -- chrom -------------------------------------------------------------- #
    p = sub.add_parser("chrom", help="Chromosome-specific k-mer filtering (MSB).")
    p.add_argument("--density", required=True,
                   help="Dir of density matrices: Chr{XX}.csv")
    p.add_argument("--output", required=True,
                   help="Output file path (single CSV: kmer, n_chrom, msb)")

    # -- filter ------------------------------------------------------------- #
    p = sub.add_parser("filter", help="Merge centro + chrom into final specific k-mer set.")
    p.add_argument("--msb", required=True,
                   help="Inter-chromosome MSB file (kmer, n_chrom, msb)")
    p.add_argument("--centro-score", required=True,
                   help="Dir with Chr{XX}.csv centromere specificity scores")
    p.add_argument("--output", required=True,
                   help="Output dir: Chr{XX}.csv specific k-mers")
    p.add_argument("--msb-ratio", type=float, default=0.05,
                   help="Keep top msb_ratio of multi-chrom kmers by MSB (default: 0.05; 0 = only n_chrom==1)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "t2t":
        t2t.run_t2t(
            t2t_dir=args.t2t_dir,
            out_dir=args.output,
            k=args.k,
        )
    elif args.command == "centro":
        centro.run_centro(
            cen=args.cen,
            t2t=args.t2t,
            chrom_lib=args.chrom_lib,
            output=args.output,
            score_threshold=args.score_threshold,
        )
    elif args.command == "chrom":
        chrom.run_chrom(
            density=args.density,
            output=args.output,
        )
    elif args.command == "filter":
        filter.run_filter(
            msb=args.msb,
            centro_score=args.centro_score,
            output=args.output,
            msb_ratio=args.msb_ratio,
        )


if __name__ == "__main__":
    main()
