"""Data loading and adaptive k-mer selection utilities.

AdaptiveDataFrame computes per-kmer F-values (between-group / within-group
variance ratio) for hierarchical k-mer selection at each clustering level.
"""

import gzip
import re
from collections import defaultdict
from io import StringIO

import jax.numpy as jnp
import numpy as np
import polars as pl


class AdaptiveDataFrame:
    """Wrap a polars DataFrame with F-value based adaptive k-mer selection."""

    def __init__(self, df: pl.DataFrame, feature_col: str) -> None:
        self._df = df
        self._feature_col = feature_col

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def feature_col(self) -> str:
        return self._feature_col

    @property
    def sample_cols(self) -> list[str]:
        return [col for col in self._df.columns if col != self._feature_col]

    def _df2grouped_dict(self, group_list: list) -> dict:
        """Map group_id -> [sample column names]."""
        grouped_dict = defaultdict(list)
        for k, v in zip(group_list, self.sample_cols):
            grouped_dict[k].append(v)
        return grouped_dict

    def _f_value(self, group_list: list) -> np.ndarray:
        """Compute F-value = MSB / MSW for each k-mer."""
        df_raw = self._df
        group_dict = self._df2grouped_dict(group_list)

        group_stats = {}
        for group_id, member in group_dict.items():
            group_samples = df_raw.select(pl.col(member)).to_numpy()
            n_members = group_samples.shape[1]
            group_stats[group_id] = {
                "mean": np.mean(group_samples, axis=1),
                # Guard ddof=1 against single-member groups (n-1=0 -> NaN).
                # A single-member group has no within-group variance.
                "var": (np.var(group_samples, axis=1, ddof=1)
                        if n_members > 1
                        else np.zeros(group_samples.shape[0])),
                "num": n_members,
            }

        all_sample_num = df_raw.width - 1
        group_num = len(group_dict)
        msw = np.zeros(len(df_raw))
        msb = np.zeros(len(df_raw))
        mean_all = np.mean(
            df_raw.select(self.sample_cols).to_numpy(), axis=1
        )
        for stats in group_stats.values():
            msw = msw + (stats["var"] * stats["num"])
            msb = msb + stats["num"] * (mean_all - stats["mean"]) ** 2

        msw = msw / max(all_sample_num - group_num, 1)
        msb = msb / max(group_num - 1, 1)

        return msb / (msw + 0.0001)

    def get_haps(self, group_list: list, specific_features: list,
                 top_pct: float = 0.15) -> pl.DataFrame:
        """Select top k-mers by F-value and build haplotype profiles.

        Returns a DataFrame with columns: feature_col, hap_1, hap_2, ...
        where each hap_ column is the mean across samples in that group.
        """
        df_raw = self._df
        f_values = self._f_value(group_list)
        df_select = df_raw.with_columns(pl.Series(f_values).alias("f_values"))

        percent_num = 100 - top_pct * 100
        f_threshold = np.percentile(df_select["f_values"], percent_num)
        df_select = (
            df_select
            .filter(pl.col("f_values") >= f_threshold)
            .drop("f_values")
            .filter(pl.col(self._feature_col).is_in(specific_features))
        )

        group_dict = self._df2grouped_dict(group_list)
        exprs = [
            pl.mean_horizontal(pl.col(member)).alias(f"hap_{group_id}")
            for group_id, member in group_dict.items()
        ]

        return df_select.select(pl.col(self._feature_col), *exprs)


def load_ngs(sample: str, chrom: str, selected_kmers: list,
             ngs_kmers_dir: str, ngs_depth_dir: str) -> tuple[pl.DataFrame, float]:
    """Load NGS k-mer counts (filtered to selected_kmers) and per-chrom depth."""
    ngs_raw = pl.scan_csv(f"{ngs_kmers_dir}/{sample}.csv").select(pl.col(["kmer", "count"]))

    df_depth = pl.read_csv(ngs_depth_dir)
    
    depth = df_depth.filter(
        pl.col("sample") == sample,
        pl.col("chrom") == chrom
    )["depth"][0]

    ngs_select = ngs_raw.filter(pl.col("kmer").is_in(selected_kmers)).collect()
    return ngs_select, depth


def df2jnp(df_haps: pl.DataFrame, ngs_count: pl.DataFrame, depth: float):
    """Align haplotype expressions with NGS counts, convert to JAX arrays.

    Returns (hap_expr, obs_count, n_haps, col_norms) where obs_count is
    normalized by sequencing depth, hap_expr is column-L2-normalized,
    and col_norms are the pre-normalization column norms of hap_expr.
    col_norms is needed downstream to rescale model props (unit-norm
    direction weights) back to genome proportions: q_i ∝ p_i / ||H_i||.
    """
    df_align = df_haps.join(ngs_count, how="left", on="kmer").fill_null(0)

    hap_expr = jnp.array(
        df_align.select(pl.col("^hap_.*$")).to_numpy(), dtype=jnp.float32
    )
    col_norms = jnp.linalg.norm(hap_expr, axis=0)
    hap_expr = hap_expr / (col_norms[None, :] + 1e-8)

    obs_count = jnp.array(
        df_align.select(pl.col("count")).to_numpy().ravel() / depth,
        dtype=jnp.float32,
    )
    n_haps = hap_expr.shape[1]

    return hap_expr, obs_count, n_haps, col_norms
