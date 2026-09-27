"""Prepare per-sample k-mer counts from deconvolution results.

For every ``{chrom}_{sample}`` in the deconv result dir, the principal
haplotype groups at one cut level are extracted, and the k-mer component
matrix is weighted by their proportions and signal to predict per-sample
k-mer counts:

    pre_kmer_count/{chrom}_{sample}.csv   (columns: kmer, count)
    pre_kmer_count/{chrom}_{sample}.fa    (k-mer sequences for minimap2)
    pre_kmer_count/sample.list            (one {chrom}_{sample} per line)
"""

import os
from pathlib import Path

import polars as pl


def get_deconv_res(sample_name: str,
                   chrom: str,
                   res_dir: str,
                   res_level: int = 17,
                   principal_prop: float = 0.95) -> pl.DataFrame:
    """Extract the principal haplotype groups at one cut level.

    Joins the proportion table with the tree grouping column at
    *res_level*, sorts groups by descending proportion and keeps the
    leading groups whose cumulative proportion reaches *principal_prop*.

    Parameters
    ----------
    sample_name : str
        Sample name (without the chromosome prefix).
    chrom : str
        Chromosome name, e.g. ``Chr01``.
    res_dir : str
        Deconv result dir (contains ``{chrom}_{sample}_props.csv`` and
        ``{chrom}_{sample}_tree.csv``).
    res_level : int
        Cut level of the deconv tree to use (default: 17).
    principal_prop : float
        Cumulative proportion threshold for principal groups
        (default: 0.95).
    """
    res_path = Path(res_dir)

    df_props = pl.read_csv(res_path / f"{chrom}_{sample_name}_props.csv")
    level_props = df_props.filter(pl.col("cut_level") == res_level)

    df_tree = pl.read_csv(res_path / f"{chrom}_{sample_name}_tree.csv")
    level_col = [col for col in df_tree.columns if col.startswith(f"cut{res_level}")]
    level_tree = df_tree.select(pl.col(["samples"] + level_col))

    df_res = (
        level_props
        .join(level_tree, how="left", left_on="group_id", right_on=level_col)
        .sort("prop", descending=True)
        .with_columns(pl.col("prop").cum_sum().alias("props_cumsum"))
    )

    idx = df_res["props_cumsum"].search_sorted(principal_prop)
    df_principal = df_res.slice(0, idx + 1)

    return df_principal


def get_kmer_counts(df_principal: pl.DataFrame, component_dir: str) -> pl.DataFrame:
    """Predict per-sample k-mer counts from the component matrix.

    The component matrix columns of the principal groups' samples are
    weighted by the group proportions and the sample signal, then
    summed per k-mer.  K-mers with a predicted count below 1 are
    dropped.
    """
    samples = df_principal["samples"].to_list()
    chrom = df_principal["chrom"][0]
    signal = df_principal["signal"][0]
    props = df_principal["prop"].to_numpy()
    component_path = Path(component_dir)

    df_component = pl.scan_csv(component_path / f"{chrom}.csv").select(pl.col(["kmer"] + samples))
    kmers = df_component.select(pl.col("kmer"))
    component_mat = df_component.drop(pl.col("kmer")).collect().to_numpy()

    pre_counts = ((component_mat * props) * signal).sum(axis=1)

    df_kmer = (
        pl.DataFrame({
            "kmer": kmers.collect().to_series().to_list(),
            "count": pre_counts,
        })
        .group_by("kmer")
        .agg(pl.col("count").sum())
        .filter(pl.col("count") >= 1)
    )

    return df_kmer


def run_prep(res_dir: str,
             component_dir: str,
             out_dir: str,
             res_level: int = 17,
             principal_prop: float = 0.95) -> None:
    """Run the prep step for every deconv result in *res_dir*.

    Writes ``{chrom}_{sample}.csv``, ``{chrom}_{sample}.fa`` and
    ``sample.list`` into *out_dir*.
    """
    os.makedirs(out_dir, exist_ok=True)

    suffixes = ("_props.csv", "_tree.csv")

    deconv_res_list = sorted({
        f.removesuffix(suf)
        for f in os.listdir(res_dir)
        for suf in suffixes
        if f.endswith(suf)
    })

    for res in deconv_res_list:
        res_part = res.split("_")
        chrom = res_part[0]
        sample = res.removeprefix(f"{chrom}_")

        df_principal = get_deconv_res(sample, chrom, res_dir,
                                      res_level, principal_prop)
        df_kmer = get_kmer_counts(df_principal, component_dir)
        df_kmer.write_csv(f"{out_dir}/{res}.csv")

        # One fasta record per k-mer: name and sequence are identical,
        # so minimap2 hit targets can be read directly from the PAF.
        with open(f"{out_dir}/{res}.fa", "w") as handle:
            for row in df_kmer.iter_rows():
                handle.write(f">{row[0]}\n{row[0]}\n")

    with open(f"{out_dir}/sample.list", "w") as handle:
        for sample in deconv_res_list:
            handle.write(f"{sample}\n")

    print(f"[INFO] prep done: {len(deconv_res_list)} chrom_sample result(s) "
          f"written to {out_dir}")
