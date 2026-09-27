"""deconv CLI: centromere haplotype deconvolution.

Usage:
  python -m deconv run --data-root ... --output ... [--n-steps 10000] [--seed 42]
                       [--top-pct 0.1] [--sparsity-weight 0.01] [--top-k 2]
                       [--level-stride 4] [--lr 0.005]
  python -m deconv run-one --data-root ... --output ... --sample 13-65 --chrom Chr01
                       (same options; for SLURM array jobs / single-pair reruns)
"""

import argparse

from . import core


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Shared training/model hyperparameters for run and run-one."""
    p.add_argument("--distance-mat", required=True,
                   help="Directory of per-chrom distance matrices (Chr{XX}.csv)")
    p.add_argument("--specific-kmers", required=True,
                   help="Directory of per-chrom specific k-mer sets (Chr{XX}.csv)")
    p.add_argument("--density-mat", required=True,
                   help="Directory of per-chrom density matrices (Chr{XX}.csv)")
    p.add_argument("--ngs-kmers", required=True,
                   help="Directory of NGS k-mer count files ({sample}.csv)")
    p.add_argument("--ngs-depth", required=True,
                   help="Directory of NGS per-chrom depth files ({sample}.chr.stat.gz)")
    p.add_argument("--output", required=True,
                   help="Output dir for deconv results")
    p.add_argument("--n-steps", type=int, default=10000,
                   help="Training steps per tree level (default: 10000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--top-pct", type=float, default=0.1,
                   help="Top fraction of kmers by F-value to keep (default: 0.1)")
    p.add_argument("--sparsity-weight", type=float, default=0.01,
                   help="Weight for top-k sparsity penalty on props "
                   "(mass beyond top-k groups). (default: 0.01)")
    p.add_argument("--top-k", type=int, default=2,
                   help="Number of top haplotypes allowed non-zero prop "
                   "(diploid=2). (default: 2)")
    p.add_argument("--level-stride", type=int, default=4,
                   help="Keep every Nth tree cut level to skip redundant "
                   "levels; first and last levels are always kept. "
                   "1 = keep all levels. (default: 4)")
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Peak learning rate for Adam with cosine decay "
                   "to 2%%. (default: 0.005)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deconv",
        description="Centromere haplotype deconvolution via hierarchical clustering + JAX.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --------------------------------------------------------------- #
    p = sub.add_parser("run", help="Run deconvolution on all samples x chromosomes.")
    _add_common_args(p)

    # -- run-one ----------------------------------------------------------- #
    p1 = sub.add_parser("run-one",
                        help="Run deconvolution on ONE sample x chromosome pair "
                             "(for SLURM array jobs or single-pair reruns).")
    _add_common_args(p1)
    p1.add_argument("--sample", required=True,
                    help="Sample name (NGS_kmers/<sample>.tsv without .tsv)")
    p1.add_argument("--chrom", required=True,
                    help="Chromosome id, e.g. Chr01")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        core.run_deconv(
            distance_mat=args.distance_mat,
            specific_kmers=args.specific_kmers,
            density_mat=args.density_mat,
            ngs_kmers=args.ngs_kmers,
            ngs_depth=args.ngs_depth,
            output=args.output,
            n_steps=args.n_steps,
            seed=args.seed,
            top_pct=args.top_pct,
            sparsity_weight=args.sparsity_weight,
            top_k=args.top_k,
            level_stride=args.level_stride,
            learning_rate=args.lr,
        )
    elif args.command == "run-one":
        core.deconv_one(
            sample=args.sample,
            chrom=args.chrom,
            distance_mat=args.distance_mat,
            specific_kmers=args.specific_kmers,
            density_mat=args.density_mat,
            ngs_kmers=args.ngs_kmers,
            ngs_depth=args.ngs_depth,
            output=args.output,
            n_steps=args.n_steps,
            seed=args.seed,
            top_pct=args.top_pct,
            sparsity_weight=args.sparsity_weight,
            top_k=args.top_k,
            level_stride=args.level_stride,
            learning_rate=args.lr,
        )


if __name__ == "__main__":
    main()
