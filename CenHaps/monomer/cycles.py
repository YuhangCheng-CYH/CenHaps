"""Detect circular monomer units from predicted k-mer counts and NGS
read-supported k-mer paths.

Background
----------
The k-mer count table (columns: kmer, count) is produced by the prep
step (prep.py), which weights the k-mer component matrix by the
principal haplotype proportions of the deconvolution result.  The
k-mers are treated as the nodes of a directed **connection graph**
(de Bruijn style): an edge A -> B exists when  A[1:] == B[:-1]  --
i.e. the two k-mers overlap by k-1 characters and can be concatenated
into a longer sequence.  This connection graph is the primary
structure we work on.

A circular monomer unit, when tiled by its k-mers, closes back on
itself: the suffix of the last k-mer matches the prefix of the first
k-mer.  In the connection graph this appears as a **simple cycle**.
We expect monomer units to be 130-180 bp long, which translates to
cycle lengths of (monomer_len - k + 1) = 104-154 k-mers.

The connection graph contains many branching points (nodes with
out-degree > 1) because the predicted k-mer set mixes multiple
similar-but-not-identical monomer copies.  These branches create a huge
number of combinatorial cycles, most of which are chimeric (they splice
together pieces from different monomer variants).  NGS reads tell us
which branches are real: an edge that no read ever traversed is almost
certainly a chimera junction.  We use NGS support to filter cycles at
their branching points -- the places where a wrong choice creates a
chimera.

K-mer counts (predicted, possibly non-integer) are used to estimate
the copy number of each cycle: the median of the node counts on the
cycle, a robust statistic that is insensitive to outliers.

Pipeline
--------
1.  Load predicted k-mer counts from the prep-step CSV
    (pre_kmer_count/{chrom}_{sample}.csv, columns: kmer, count).
2.  Load NGS read-supported k-mer paths from minimap2 PAF alignments
    (verify.paf2verify).
3.  Build the connection (de Bruijn) graph from the CSV k-mer column.
4.  Weight every edge with NGS pair-support (reads traversing it).
5.  Prune tips iteratively (dead-end nodes) to keep the cyclic core.
6.  Enumerate simple cycles within a length window
    [min_cycle_len, max_cycle_len] using Johnson's algorithm, with a
    total cap on the number of cycles to bound memory.
7.  Filter chimeric cycles at branching points: a cycle is rejected if
    it traverses an unsupported edge (support == 0) at a node whose
    out-degree > 1 (a real branching point).  Non-branching edges are
    allowed to have zero support (gaps in read coverage).
8.  Deduplicate cycles that are cyclic rotations of each other.
9.  Estimate copy number per cycle from median k-mer count.

Outputs
-------
  {chrom}_{sample}_cycles_summary.tsv  - one row per cycle (id, n_nodes,
                                         branch stats, NGS stats,
                                         copy number)
  {chrom}_{sample}_cycles_detail.json  - full cycle node lists and
                                         per-edge support
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import polars as pl

from .verify import paf2verify


# ---------------------------------------------------------------------------
# 1. Load k-mer counts (prep-step output)
# ---------------------------------------------------------------------------
def load_kmer_counts(csv_path: Path) -> dict[str, float]:
    """Load the k-mer count table produced by the prep step.

    The CSV is written by prep.py: the k-mer component matrix weighted
    by the principal haplotype proportions of the deconvolution
    result.  Columns: kmer, count.
    """
    df = pl.read_csv(csv_path)
    counts: dict[str, float] = dict(
        zip(df["kmer"].to_list(), df["count"].cast(pl.Float64).to_list())
    )
    print(f"[INFO] Loaded {len(counts)} k-mers from {csv_path.name}")
    return counts


# ---------------------------------------------------------------------------
# 2. Build pair-support Counter from NGS paths
# ---------------------------------------------------------------------------
def build_pair_support(paths: list[list[str]]) -> Counter:
    """Count how many reads support each consecutive (u, v) k-mer pair."""
    pair_support: Counter = Counter()
    for path in paths:
        pair_support.update(zip(path, path[1:]))
    print(f"[INFO] Built pair support for {len(pair_support)} unique edges")
    return pair_support


# ---------------------------------------------------------------------------
# 3. Build the connection (de Bruijn) graph from CSV k-mers
# ---------------------------------------------------------------------------
def build_connection_graph(
    kmer_counts: dict[str, float],
    pair_support: Counter,
    k: int = 27,
) -> nx.DiGraph:
    """Construct the directed connection graph from the CSV k-mer column.

    Nodes are the k-mers.  An edge A -> B exists when A[1:] == B[:-1]
    (k-1 overlap).  Edge attribute 'support' is the number of NGS reads
    that traversed this exact pair (0 if no read supports it).
    """
    kmers = list(kmer_counts.keys())
    # Index k-mers by their (k-1)-prefix for O(n) neighbour lookup
    prefix_idx: dict[str, list[str]] = defaultdict(list)
    for km in kmers:
        prefix_idx[km[: k - 1]].append(km)

    g = nx.DiGraph()
    for km in kmers:
        g.add_node(km, count=kmer_counts[km])
        suffix = km[1:]  # last k-1 characters
        for nxt in prefix_idx.get(suffix, []):
            if nxt == km:
                continue
            s = pair_support.get((km, nxt), 0)
            g.add_edge(km, nxt, support=s)

    ngs_supported = sum(1 for _, _, d in g.edges(data=True) if d["support"] > 0)
    print(f"[INFO] Connection graph: {g.number_of_nodes()} nodes, "
          f"{g.number_of_edges()} edges "
          f"({ngs_supported} NGS-supported)")
    return g


# ---------------------------------------------------------------------------
# 4. Prune tips iteratively
# ---------------------------------------------------------------------------
def prune_tips(g: nx.DiGraph) -> nx.DiGraph:
    """Iteratively remove dead-end nodes (in-deg 0 or out-deg 0).

    This trims linear branches and leaves the cyclic core, reducing the
    graph size before cycle enumeration.
    """
    rounds = 0
    while True:
        dead = [n for n in g.nodes()
                if g.in_degree(n) == 0 or g.out_degree(n) == 0]
        if not dead:
            break
        g.remove_nodes_from(dead)
        rounds += 1
    print(f"[INFO] Tip pruning done in {rounds} round(s); "
          f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges remain")
    return g


# ---------------------------------------------------------------------------
# 5. Enumerate simple cycles within a length window
# ---------------------------------------------------------------------------
def find_cycles(g: nx.DiGraph,
                min_len: int = 104,
                max_len: int = 154,
                max_cycles: int = 5000) -> list[list[str]]:
    """Find simple cycles whose length is within [min_len, max_len].

    Uses nx.simple_cycles (Johnson's algorithm) with a length bound of
    *max_len*, then keeps only cycles of length >= min_len.  A safety
    cap *max_cycles* bounds memory usage on graphs with combinatorially
    many cycles.

    Parameters
    ----------
    g : nx.DiGraph
        Pruned directed connection graph.
    min_len, max_len : int
        Cycle length (number of k-mer nodes) window.  For monomer units
        of 130-180 bp and k=27, this is 104-154.
    max_cycles : int
        Safety cap on the total number of cycles to collect.
    """
    cycles: list[list[str]] = []
    try:
        for cyc in nx.simple_cycles(g, length_bound=max_len):
            if len(cyc) >= min_len:
                cycles.append(cyc)
            if len(cycles) >= max_cycles:
                print(f"[WARN] Hit max_cycles cap ({max_cycles}); "
                      f"stopping enumeration early.",
                      file=sys.stderr)
                break
    except Exception as exc:
        print(f"[WARN] simple_cycles raised {exc}; returning empty list",
              file=sys.stderr)
    print(f"[INFO] Found {len(cycles)} raw cycles in length window "
          f"[{min_len}, {max_len}] (cap={max_cycles})")
    return cycles


# ---------------------------------------------------------------------------
# 6. Filter chimeric cycles at branching points
# ---------------------------------------------------------------------------
def filter_chimeric_cycles(g: nx.DiGraph,
                           cycles: list[list[str]]) -> list[list[str]]:
    """Reject cycles that take an unsupported edge at a branching point.

    A **branching point** is a node whose out-degree in *g* is > 1 --
    i.e. there is more than one possible next k-mer, so the cycle had to
    make a choice.  If the cycle's chosen edge at that point has zero
    NGS support (no read ever traversed it), the cycle is almost
    certainly a chimera splicing together pieces from different monomer
    variants, and is rejected.

    Edges at non-branching points (out-degree == 1) are allowed to have
    zero support -- a gap in read coverage on a linear stretch is
    normal in complex repeat regions.
    """
    kept: list[list[str]] = []
    stats_total_branches = 0
    stats_rejected_branches = 0
    for cyc in cycles:
        n = len(cyc)
        ok = True
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            is_branch = g.out_degree(u) > 1
            if is_branch:
                stats_total_branches += 1
                data = g.get_edge_data(u, v)
                s = data.get("support", 0) if data else 0
                if s == 0:
                    # Unsupported edge at a branching point -> chimera.
                    ok = False
                    stats_rejected_branches += 1
                    break
        if ok:
            kept.append(cyc)
    print(f"[INFO] Kept {len(kept)} / {len(cycles)} cycles after "
          f"branch-point chimera filtering "
          f"({stats_rejected_branches}/{stats_total_branches} "
          f"branch choices rejected)")
    return kept


# ---------------------------------------------------------------------------
# 7. Score a cycle by NGS edge support (for reporting)
# ---------------------------------------------------------------------------
def cycle_edge_support(g: nx.DiGraph,
                       cycle: list[str]) -> tuple[int, float, int, float]:
    """Return (min_support, mean_support, n_supported_edges,
    support_ratio) for a cycle.

    n_supported_edges counts edges with support >= 1 (i.e. traversed
    by at least one NGS read).  support_ratio is that count divided
    by the cycle length.  These are reported for information; the
    chimera filter uses branching-point logic, not these global stats.
    """
    supports = []
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        data = g.get_edge_data(u, v)
        if data is not None:
            supports.append(data.get("support", 0))
    if not supports:
        return 0, 0.0, 0, 0.0
    n_supported = sum(1 for s in supports if s >= 1)
    ratio = n_supported / len(supports)
    return min(supports), sum(supports) / len(supports), n_supported, ratio


def cycle_branch_stats(g: nx.DiGraph,
                       cycle: list[str]) -> tuple[int, int, int]:
    """Return (n_branch_points, n_supported_branches, n_unsupported_branches).

    A branch point is a cycle node whose out-degree in *g* is > 1.
    """
    n_bp = 0
    n_sup = 0
    n_unsup = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if g.out_degree(u) > 1:
            n_bp += 1
            data = g.get_edge_data(u, v)
            s = data.get("support", 0) if data else 0
            if s > 0:
                n_sup += 1
            else:
                n_unsup += 1
    return n_bp, n_sup, n_unsup


# ---------------------------------------------------------------------------
# 8. Deduplicate cycles that are rotations of each other
# ---------------------------------------------------------------------------
def dedup_cycles(cycles: list[list[str]]) -> list[list[str]]:
    """Remove duplicate cycles that are cyclic rotations of each other."""
    seen: set[tuple] = set()
    unique: list[list[str]] = []
    for cyc in cycles:
        n = len(cyc)
        if n == 0:
            continue
        rotations = [tuple(cyc[i:] + cyc[:i]) for i in range(n)]
        canon = min(rotations)
        if canon not in seen:
            seen.add(canon)
            unique.append(cyc)
    print(f"[INFO] {len(unique)} unique cycles after deduplication")
    return unique


# ---------------------------------------------------------------------------
# 9. Estimate copy number from k-mer counts
# ---------------------------------------------------------------------------
def estimate_copy_number(cycle: list[str],
                         kmer_counts: dict[str, float]) -> float:
    """Estimate the copy number of a cycle using the median k-mer count.

    K-mer counts are predicted copy abundances (not necessarily
    integers), so the median of the node counts on the cycle gives a
    robust estimate that is insensitive to individual outlier k-mers.
    """
    vals = [kmer_counts.get(k, 0.0) for k in cycle]
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        median = (vals[mid - 1] + vals[mid]) / 2.0
    else:
        median = vals[mid]
    return round(median, 4)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_cycles(chrom_sample: str,
               kmer_count_dir: Path,
               ngs_verify_dir: Path,
               out_dir: Path,
               kmer_len: int = 27,
               align_threshold: float = 0.8,
               min_cycle_len: int = 104,
               max_cycle_len: int = 154,
               max_cycles: int = 5000) -> None:
    """Run the full cycle-detection pipeline for one {chrom}_{sample}.

    Inputs (by convention, resolved from *chrom_sample*):
      {kmer_count_dir}/{chrom_sample}.csv    - prep-step k-mer counts
      {ngs_verify_dir}/{chrom_sample}_1.paf  - minimap2 alignments, mate 1
      {ngs_verify_dir}/{chrom_sample}_2.paf  - minimap2 alignments, mate 2
    """
    count_path = kmer_count_dir / f"{chrom_sample}.csv"
    paf_1 = ngs_verify_dir / f"{chrom_sample}_1.paf"
    paf_2 = ngs_verify_dir / f"{chrom_sample}_2.paf"

    for path in (count_path, paf_1, paf_2):
        if not path.exists():
            print(f"[ERROR] Missing input {path}; skipping {chrom_sample}",
                  file=sys.stderr)
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load predicted k-mer counts (prep-step output)
    kmer_counts = load_kmer_counts(count_path)

    # Step 2: load NGS read-supported k-mer paths from PAF alignments
    paths = paf2verify(paf_1, paf_2, align_threshold, kmer_len)

    # Step 3: build NGS pair support
    pair_support = build_pair_support(paths)

    # Step 4: build connection graph from CSV k-mers, weighted by NGS support
    g = build_connection_graph(kmer_counts, pair_support, k=kmer_len)

    # Step 5: prune tips
    g = prune_tips(g)
    if g.number_of_nodes() == 0:
        print(f"[ERROR] Graph is empty after tip pruning; "
              f"skipping {chrom_sample}", file=sys.stderr)
        return

    # Step 6: find cycles in the monomer-length window
    cycles = find_cycles(g,
                         min_len=min_cycle_len,
                         max_len=max_cycle_len,
                         max_cycles=max_cycles)
    if not cycles:
        print(f"[WARN] No cycles found for {chrom_sample} in the length "
              f"window [{min_cycle_len}, {max_cycle_len}].", file=sys.stderr)
        return

    # Step 7: filter chimeric cycles at branching points
    cycles = filter_chimeric_cycles(g, cycles)

    # Step 8: deduplicate
    cycles = dedup_cycles(cycles)

    # Step 9: estimate copy number & write outputs
    summary_path = out_dir / f"{chrom_sample}_cycles_summary.tsv"
    detail_path = out_dir / f"{chrom_sample}_cycles_detail.json"

    with open(summary_path, "w", newline="") as tsv_fh:
        writer = csv.writer(tsv_fh, delimiter="\t")
        writer.writerow([
            "cycle_id", "n_nodes",
            "n_branch_points", "n_supported_branches",
            "n_unsupported_branches",
            "min_edge_support", "mean_edge_support",
            "n_supported_edges", "support_ratio",
            "copy_number", "first_node"
        ])
        detail_records = []
        for idx, cyc in enumerate(cycles):
            n_bp, n_sup_b, n_unsup_b = cycle_branch_stats(g, cyc)
            min_s, mean_s, n_sup, ratio = cycle_edge_support(g, cyc)
            cn = estimate_copy_number(cyc, kmer_counts)
            writer.writerow([
                idx, len(cyc),
                n_bp, n_sup_b, n_unsup_b,
                min_s, round(mean_s, 4),
                n_sup, round(ratio, 4),
                cn, cyc[0]
            ])
            # Per-edge support detail (compact list of integers)
            edge_supports = []
            n = len(cyc)
            for i in range(n):
                u = cyc[i]
                v = cyc[(i + 1) % n]
                data = g.get_edge_data(u, v)
                s = data.get("support", 0) if data else 0
                edge_supports.append(s)
            detail_records.append({
                "cycle_id": idx,
                "nodes": cyc,
                "n_nodes": len(cyc),
                "n_branch_points": n_bp,
                "n_supported_branches": n_sup_b,
                "n_unsupported_branches": n_unsup_b,
                "min_edge_support": min_s,
                "mean_edge_support": round(mean_s, 4),
                "n_supported_edges": n_sup,
                "support_ratio": round(ratio, 4),
                "copy_number": cn,
                "edge_supports": edge_supports,
            })

    with open(detail_path, "w") as jf:
        json.dump(detail_records, jf, indent=2)

    print(f"[INFO] Wrote {len(cycles)} cycles for {chrom_sample}:")
    print(f"       Summary : {summary_path}")
    print(f"       Details : {detail_path}")


def run_cycles_all(kmer_count_dir: Path,
                   ngs_verify_dir: Path,
                   out_dir: Path,
                   chrom_samples: list[str] | None = None,
                   kmer_len: int = 27,
                   align_threshold: float = 0.8,
                   min_cycle_len: int = 104,
                   max_cycle_len: int = 154,
                   max_cycles: int = 5000) -> None:
    """Run cycle detection for one or all samples.

    When *chrom_samples* is None the list is read from
    ``{kmer_count_dir}/sample.list`` (one ``{chrom}_{sample}`` per
    line, written by the prep step).
    """
    if chrom_samples is None:
        list_path = kmer_count_dir / "sample.list"
        with open(list_path) as fh:
            chrom_samples = [line.strip() for line in fh if line.strip()]

    for chrom_sample in chrom_samples:
        print(f"[INFO] === {chrom_sample} ===")
        run_cycles(
            chrom_sample=chrom_sample,
            kmer_count_dir=kmer_count_dir,
            ngs_verify_dir=ngs_verify_dir,
            out_dir=out_dir,
            kmer_len=kmer_len,
            align_threshold=align_threshold,
            min_cycle_len=min_cycle_len,
            max_cycle_len=max_cycle_len,
            max_cycles=max_cycles,
        )
