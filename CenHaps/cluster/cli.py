"""cluster CLI: iterative NMF clustering.

Usage:
  python -m cluster nmf --density-mat ... --output ... [--n-iters 600] [--n-jobs -1]
"""

import argparse

from . import nmf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster",
        description="Iterative NMF clustering for centromere k-mer matrices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- nmf --------------------------------------------------------------- #
    p = sub.add_parser("nmf", help="Iterative NMF distance matrix.")
    p.add_argument("--density-mat", required=True,
                   help="Dir of density matrices: named like Chr{XX}.csv ")
    p.add_argument("--output", required=True,
                   help="Output dir of distance matrices: named like Chr{XX}.csv")
    p.add_argument("--n-iters", type=int, default=600,
                   help="Number of iterations (default: 900)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Number of parallel workers (default: -1 = all cores)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "nmf":
        nmf.run_nmf(
            density_mat=args.density_mat,
            output=args.output,
            n_iters=args.n_iters,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()

