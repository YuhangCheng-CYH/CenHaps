"""Extract read-supported k-mer paths from minimap2 PAF alignments.

Reads are aligned to the k-mer fasta produced by the prep step (one
record per k-mer, name == sequence).  Consecutive k-mer hits on a read
are collapsed into fragments, each fragment is re-sliced into k-mers,
and the resulting k-mer lists are the verify paths used to weight the
connection graph in ``cycles.py``.
"""

from pathlib import Path

import polars as pl


def get_seq_kmer(seq: str, size: int, step: int = 1) -> list[str]:
    """Slide a window of *size* over *seq* and return every k-mer."""
    return [seq[i:i + size] for i in range(0, len(seq) - size + 1, step)]


def _scan_paf(paf_path: Path, strand: str) -> pl.LazyFrame:
    """Select the PAF columns needed for fragment assembly.

    Keeps only hits on the requested strand.  For reverse-strand hits
    minimap2 reports query coordinates on the reverse complement;
    they are remapped onto the read's forward orientation so both
    strands share the same downstream fragment-assembly logic.
    """
    lf = (
        pl.scan_csv(paf_path, has_header=False, separator="\t")
        .select(
            q_name=pl.col("column_1"),
            q_seq_len=pl.col("column_2"),
            q_start=pl.col("column_3"),
            q_end=pl.col("column_4"),
            strand=pl.col("column_5"),
            t_name=pl.col("column_6"),
            t_seq_len=pl.col("column_7"),
        )
        .filter(pl.col("strand") == strand)
    )
    if strand == "-":
        lf = lf.with_columns(
            q_start=pl.col("q_seq_len") - pl.col("q_end"),
            q_end=pl.col("q_seq_len") - pl.col("q_start"),
        )
    return lf


def _read_fragments(paf_path: Path,
                    strand: str,
                    threshold: float,
                    k: int) -> pl.DataFrame:
    """Collapse consecutive k-mer alignments on each read into fragments.

    A read is kept only when the number of its k-mer hits reaches
    ``(read_len - kmer_len) * threshold`` -- i.e. the read is tiled by
    k-mer alignments over (at least) that fraction of its span.  Hits
    on the same read whose query starts differ by more than ``k`` start
    a new fragment; otherwise the new k-mer extends the current
    fragment by the last ``gap`` characters of its sequence.
    """
    return (
        _scan_paf(paf_path, strand)
        .filter(
            pl.len().over("q_name")
            >= ((pl.col("q_seq_len") - pl.col("t_seq_len")) * threshold)
        )
        .sort(["q_name", "q_start"])
        .with_columns(
            gap=pl.col("q_start") - pl.col("q_start").shift(1).over("q_name"),
        )
        .with_columns(
            tail=pl.when(pl.col("gap").is_between(1, k))
                .then(pl.col("t_name").str.slice(-pl.col("gap")))
                .otherwise(None),
        )
        .with_columns(
            new_frag=(pl.col("gap") > k).fill_null(True)
        )
        .with_columns(frag_id=pl.col("new_frag").cum_sum().over("q_name"))
        .group_by(["q_name", "frag_id"])
        .agg(
            q_from=pl.col("q_start").min(),
            q_to=pl.col("q_end").max(),
            n_kmers=pl.len(),
            kmer_path=pl.col("t_name"),
            frag_seq=pl.concat_str(
                [
                    pl.col("t_name").first(),
                    pl.col("tail").drop_nulls().str.join(""),
                ],
                ignore_nulls=True,
            ),
        )
        .sort(["q_name", "frag_id"])
        .collect()
    )


def paf2verify(paf_1: Path,
               paf_2: Path,
               threshold: float,
               k: int) -> list[list[str]]:
    """Assemble read-supported k-mer paths from paired-end PAF files.

    Both strands of both mate files are processed, fragments supported
    by a single k-mer hit are dropped, and every remaining fragment
    sequence is re-sliced into k-mers of length ``k``.

    Parameters
    ----------
    paf_1, paf_2 : Path
        minimap2 PAF files of mate 1 / mate 2 against the k-mer fasta.
    threshold : float
        Minimum fraction of the read span covered by k-mer hits.
    k : int
        K-mer length; must match the ``-k`` value used by minimap2.
    """
    fragments = (
        pl.concat([
            _read_fragments(paf_1, "+", threshold, k),
            _read_fragments(paf_1, "-", threshold, k),
            _read_fragments(paf_2, "+", threshold, k),
            _read_fragments(paf_2, "-", threshold, k),
        ])
        .filter(pl.col("n_kmers") > 1)
    )

    verify_path_list = [get_seq_kmer(frag_seq, k) for frag_seq in fragments["frag_seq"]]
    print(f"[INFO] Loaded {len(verify_path_list)} read-supported paths "
          f"from {paf_1.name}, {paf_2.name}")
    return verify_path_list
