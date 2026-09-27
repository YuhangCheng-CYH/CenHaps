"""Subcommand ``centro``: centromere-specific k-mer filtering.

For each chromosome, compare centromere k-mer counts against whole-genome
(t2t) counts, compute a specificity score, and output the ratio of
samples passing the score threshold.

Input:
  - cen kmer counts:  {cen}/Chr{XX}_{samplename}.csv   (kmer, count, ...extra columns ignored)
  - t2t kmer counts:  {t2t}/Chr{XX}_{samplename}.csv   (kmer, count, ...extra columns ignored)
  - chrom kmer lib:   {chrom_lib}/Chr{XX}.csv          (kmer)
Output:
  - specificity score: {output}/Chr{XX}.csv
    (kmer, get_samples, passed_samples, ratio)
"""

import os

import polars as pl

from . import CHROMS


def run_centro(cen: str, t2t: str, chrom_lib: str, output: str,
               score_threshold: float = 0.9) -> None:
    os.makedirs(output, exist_ok=True)

    cen_list = os.listdir(cen)

    for chrom in CHROMS:
        kmer_lib = pl.scan_csv(os.path.join(chrom_lib, f"{chrom}.csv"))
        passed_samples = kmer_lib
        all_samples = kmer_lib

        selected_files = [
            f for f in cen_list
            if f.startswith(chrom) and f.endswith(".csv")
        ]

        for sample_file in selected_files:
            sample_name = sample_file.removesuffix(".csv")

            cen_path = os.path.join(cen, sample_file)
            t2t_path = os.path.join(t2t, sample_file)


            # Keep only kmer/count
            cen_count = pl.scan_csv(cen_path).select(
                pl.col("kmer"), pl.col("count").alias("cen_count")
            )
            t2t_count = pl.scan_csv(t2t_path).select(
                pl.col("kmer"), pl.col("count").alias("tol_count")
            )

            # Align cen vs t2t counts, keep kmers present in t2t
            df_align = cen_count.join(t2t_count, on="kmer", how="left").fill_null(0)
            df_align = df_align.filter(pl.col("tol_count") > 0)

            # outcen = kmer count outside centromere (clipped to >= 0)
            # score  = (cen_count + 1) / (cen_count + outcen + 1)
            df_align = df_align.with_columns(
                (pl.col("tol_count") - pl.col("cen_count")).clip(lower_bound=0).alias("outcen")
            ).with_columns(
                ((pl.col("cen_count") + 1) / (pl.col("cen_count") + pl.col("outcen") + 1)).alias("score")
            )

            # Centromere-specific kmers (score >= threshold)
            df_centro_spec = df_align.filter(
                pl.col("score") >= score_threshold
            ).select(pl.col("kmer")).with_columns(pl.lit(1).alias(sample_name))

            passed_samples = passed_samples.join(df_centro_spec, on="kmer", how="left").fill_null(0)
            passed_samples = passed_samples.select(
                pl.col("kmer"),
                pl.sum_horizontal(pl.all().exclude("kmer").alias("passed_samples")),
            )

            # All kmers present in centromere (cen_count > 0)
            df_get = df_align.filter(
                pl.col("cen_count") > 0
            ).select(pl.col("kmer")).with_columns(pl.lit(1).alias(sample_name))

            all_samples = all_samples.join(df_get, on="kmer", how="left").fill_null(0)
            all_samples = all_samples.select(
                pl.col("kmer"),
                pl.sum_horizontal(pl.all().exclude("kmer").alias("get_samples")),
            )

        # ratio = passed_samples / get_samples
        df_ratio = all_samples.join(passed_samples, on="kmer", how="left"
            ).with_columns(
                (pl.col("passed_samples") / pl.col("get_samples")).alias("ratio")
            ).collect()
        df_ratio.write_csv(os.path.join(output, f"{chrom}.csv"))
