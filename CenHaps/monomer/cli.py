"""monomer CLI: circular centromere monomer unit reconstruction.

Usage:
  python -m monomer prep   --res-dir ... --component-dir ... --out-dir ...
  python -m monomer cycles --kmer-count-dir ... --ngs-verify-dir ... --out-dir ...
"""

import argparse
from pathlib import Path

from . import cycles, prep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monomer",
        description="Reconstruct circular centromere monomer units from "
                    "deconvolution results and NGS read support.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- prep -------------------------------------------------------------- #
    p = sub.add_parser("prep",
                       help="Deconv results -> per-sample k-mer counts.")
    p.add_argument("--res-dir", required=True,
                   help="Deconv result dir (contains "
                        "{chrom}_{sample}_props.csv / _tree.csv)")
    p.add_argument("--component-dir", required=True,
                   help="K-mer component matrix dir (contains {chrom}.csv)")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for k-mer counts (pre_kmer_count)")
    p.add_argument("--res-level", type=int, default=17,
                   help="Cut level of the deconv tree to use (default: 17)")
    p.add_argument("--principal-prop", type=float, default=0.95,
                   help="Cumulative proportion of principal groups to keep "
                        "(default: 0.95)")

    # -- cycles ------------------------------------------------------------ #
    p = sub.add_parser("cycles",
                       help="Detect circular monomer units.")
    p.add_argument("--kmer-count-dir", required=True,
                   help="K-mer count dir from the prep step (pre_kmer_count)")
    p.add_argument("--ngs-verify-dir", required=True,
                   help="PAF alignment dir (ngs_verify)")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for cycle results (monomer_res)")
    p.add_argument("--chrom-sample", default=None,
                   help="Process a single {chrom}_{sample}; "
                        "default: all entries in sample.list")
    p.add_argument("--kmer-len", type=int, default=27,
                   help="K-mer length k; must match the minimap2 -k value "
                        "(default: 27)")
    p.add_argument("--align-threshold", type=float, default=0.8,
                   help="Minimum fraction of the read span covered by "
                        "k-mer alignments (default: 0.8)")
    p.add_argument("--min-cycle-len", type=int, default=104,
                   help="Minimum cycle length in k-mers "
                        "(monomer 130 bp -> 104)")
    p.add_argument("--max-cycle-len", type=int, default=154,
                   help="Maximum cycle length in k-mers "
                        "(monomer 180 bp -> 154)")
    p.add_argument("--max-cycles", type=int, default=5000,
                   help="Safety cap on total cycles to collect per sample "
                        "(bounds memory)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prep":
        prep.run_prep(
            res_dir=args.res_dir,
            component_dir=args.component_dir,
            out_dir=args.out_dir,
            res_level=args.res_level,
            principal_prop=args.principal_prop,
        )
    elif args.command == "cycles":
        chrom_samples = [args.chrom_sample] if args.chrom_sample else None
        cycles.run_cycles_all(
            kmer_count_dir=Path(args.kmer_count_dir),
            ngs_verify_dir=Path(args.ngs_verify_dir),
            out_dir=Path(args.out_dir),
            chrom_samples=chrom_samples,
            kmer_len=args.kmer_len,
            align_threshold=args.align_threshold,
            min_cycle_len=args.min_cycle_len,
            max_cycle_len=args.max_cycle_len,
            max_cycles=args.max_cycles,
        )


if __name__ == "__main__":
    main()
