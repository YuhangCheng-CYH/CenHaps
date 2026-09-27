"""Core deconvolution: per-sample per-chromosome haplotype inference.

Iterates through the hierarchical clustering tree; at each level:
  1. Select adaptive k-mers via F-value
  2. Build haplotype expression profiles (group means)
  3. Train a JAX model to fit NGS observed counts
  4. Propagate inferred proportions to the next level as priors
"""

import os

import jax
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster

from . import CHROMS
from .data import AdaptiveDataFrame, df2jnp, load_ngs
from .model import training_model
from .tree import get_merges_map, load_tree, map2group_assign, stride_levels


def deconv_one(sample: str, chrom: str, output: str,
               distance_mat: str, specific_kmers_dir: str, density_mat: str,
               ngs_kmers: str, ngs_depth: str,
               n_steps: int = 10000, seed: int = 42,
               top_pct: float = 0.1, sparsity_weight: float = 0.01,
               top_k: int = 2, level_stride: int = 1,
               learning_rate: float = 5e-3) -> None:
    """Run deconvolution for a single sample x chromosome pair."""
    # Load reference data
    specific_kmers = pl.read_csv(
        f"{specific_kmers_dir}/{chrom}.csv"
    )["kmer"].to_list()
    df_ref = pl.read_csv(f"{density_mat}/{chrom}.csv")
    adap_ref = AdaptiveDataFrame(df=df_ref, feature_col="kmer")

    # Build clustering tree; skip redundant levels (first/last always kept)
    Z, cut_heights = load_tree(f"{distance_mat}/{chrom}.csv")
    cut_heights = stride_levels(cut_heights, level_stride)
    merges_map = get_merges_map(Z=Z, cut_heights=cut_heights)

    # Iterate through tree levels
    previous_prop_list = [()]
    # Tree file: sample -> group_id at each cut level
    tree_dict = {"samples": df_ref.columns[1:]}
    # Structured results for long-format props output
    level_results = []

    for cut in range(len(cut_heights)):
        cut_h = cut_heights[cut]
        merge_map = merges_map[cut]
        previous_prop = previous_prop_list[cut]

        group_assign = () if not merge_map else map2group_assign(merge_map)

        group_list = fcluster(Z, t=cut_h, criterion="distance")
        tree_dict[f"cut{cut}_h{cut_h:.4f}"] = group_list

        df_haps = adap_ref.get_haps(
            group_list=group_list,
            specific_features=specific_kmers,
            top_pct=top_pct,
        )
        selected_kmers = df_haps["kmer"].to_list()
        ngs_count, depth = load_ngs(
            sample, chrom, selected_kmers,
            ngs_kmers_dir=ngs_kmers,
            ngs_depth_dir=ngs_depth,
        )
        hap_expr, obs_count, n_haps, col_norms = df2jnp(
            df_haps, ngs_count, depth
        )

        # Extract group IDs in column order (matches prop index order)
        hap_cols = [c for c in df_haps.columns if c.startswith("hap_")]
        group_ids = [int(c.split("_")[1]) for c in hap_cols]

        _, final_props, final_signal, final_loss = training_model(
            hap_expr, obs_count, n_haps,
            group_assign, previous_prop, seed, n_steps,
            sparsity_weight=sparsity_weight, top_k=top_k,
            learning_rate=learning_rate,
        )
        props_raw = [float(p) for p in np.array(final_props)]
        previous_prop_list.append(tuple(props_raw))

        # Rescale model props (unit-norm direction weights) back to genome
        # proportions: q_i ∝ p_i / ||H_i||, renormalized to sum to 1.
        # The L2 column normalization in df2jnp improves optimization
        # geometry but distorts proportion semantics: high-norm groups
        # are inflated in p-space, which would bias the 95% cumsum
        # evaluation toward calling heterozygous samples homozygous.
        # NOTE: the RAW p (not q) is propagated as the next level's
        # prior — p is the model's native space, and column-norm ratios
        # are roughly preserved across levels (same haplotypes).
        col_norms_np = np.array(col_norms, dtype=np.float64)
        props_np = np.array(props_raw, dtype=np.float64)
        q = props_np / (col_norms_np + 1e-8)
        props_q = (q / q.sum()).tolist()

        # Compute normalized entropy for diagnostics (on genome proportions)
        safe = np.array(props_q, dtype=np.float64) + 1e-8
        norm_ent = float(
            -np.sum(safe * np.log(safe)) / np.log(len(props_q))
        )

        level_results.append({
            "cut_level": cut,
            "cut_height": float(cut_h),
            "n_haps": n_haps,
            "n_kmers": len(selected_kmers),
            "group_ids": group_ids,
            "props": props_q,
            "props_raw": props_raw,
            "loss": float(final_loss),
            "entropy": norm_ent,
            "signal": float(final_signal),
        })

    # Write tree CSV: sample x group assignment per cut level
    os.makedirs(output, exist_ok=True)
    pl.DataFrame(tree_dict).write_csv(
        f"{output}/{chrom}_{sample}_tree.csv"
    )

    # Write props as long-format CSV with full metadata.
    # Each row = one (cut_level, group_id) pair, explicitly linkable
    # to the tree file via cut_level + group_id.
    # "prop" holds genome proportions (q, rescaled); "prop_raw" holds
    # the model's native unit-norm direction weights (p) for diagnostics.
    prop_records = []
    for lr in level_results:
        for gid, prop, prop_raw in zip(
            lr["group_ids"], lr["props"], lr["props_raw"]
        ):
            prop_records.append({
                "sample": sample,
                "chrom": chrom,
                "cut_level": lr["cut_level"],
                "cut_height": lr["cut_height"],
                "n_haps": lr["n_haps"],
                "n_kmers": lr["n_kmers"],
                "group_id": gid,
                "prop": prop,
                "prop_raw": prop_raw,
                "loss": lr["loss"],
                "entropy": lr["entropy"],
                "signal": lr["signal"],
            })
    pl.DataFrame(prop_records).write_csv(
        f"{output}/{chrom}_{sample}_props.csv"
    )

    jax.clear_caches()


def run_deconv(output: str, distance_mat: str, specific_kmers: str,
               density_mat: str, ngs_kmers: str, ngs_depth: str,
               n_steps: int = 10000, seed: int = 42, top_pct: float = 0.1,
               sparsity_weight: float = 0.01, top_k: int = 2,
               level_stride: int = 1, learning_rate: float = 5e-3) -> None:
    """Run deconvolution for all samples x all chromosomes."""
    os.makedirs(output, exist_ok=True)

    ngs_files = os.listdir(ngs_kmers)
    samples = [f.removesuffix(".csv") for f in ngs_files if f.endswith(".csv")]

    for sample in samples:
        for chrom in CHROMS:
            print(f"Deconvolving {sample}-{chrom}")
            deconv_one(sample, chrom, output,
                       distance_mat, specific_kmers, density_mat,
                       ngs_kmers, ngs_depth,
                       n_steps=n_steps, seed=seed,
                       top_pct=top_pct, sparsity_weight=sparsity_weight,
                       top_k=top_k, level_stride=level_stride,
                       learning_rate=learning_rate)
