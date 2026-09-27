# CenHaps

Centromere haplotype deconvolution and circular monomer unit reconstruction pipeline for T2T (Telomere-to-Telomere) genome assemblies.

CenHaps takes T2T genome assemblies and NGS reads as input, and produces centromere-specific k-mer sets, sample-level haplotype compositions, and circular monomer unit sequences. The pipeline is organized into five independent modules that can be run sequentially or standalone.

## Pipeline Overview

```
                        ┌─────────┐
   T2T FASTA ──────────►│ kmerize │──► density matrix
   cen BED + FASTA      └─────────┘
                                │
                                ▼
                       ┌──────────────┐
                       │ kmer_filter  │──► specific k-mer set
                       └──────────────┘
   density matrix +          │
   specific k-mers           ▼
                      ┌─────────┐
                      │ cluster │──► distance matrix
                      └─────────┘
   distance matrix +         │
   specific k-mers +         ▼
   NGS k-mer counts   ┌─────────┐
                       │ deconv  │──► haplotype proportions
                       └─────────┘
   deconv results +          │
   density matrix +          ▼
   NGS reads         ┌─────────┐
                    │ monomer │──► circular monomer units
                    └─────────┘
```

## Modules

### 1. `kmerize` — HOR array k-mer statistics

Extracts centromere HOR array sequences from T2T assemblies and builds per-chromosome k-mer density matrices.

- **`extract`** (shell): Slice centromere sequences from T2T FASTA using BED files + `seqkit`.
- **`kmer`**: Generate per-sample k-mer counts and chromosome k-mer libraries from HOR array FASTA.
- **`matrix`**: Build the per-chromosome k-mer density matrix (kmer × sample).

```bash
# Extract centromere sequences (requires seqkit)
bash kmerize/extract_hor.sh

# Generate k-mer counts + chromosome libraries
python -m kmerize kmer \
    --k 27 \
    --hor ${DATA}/hor \
    --o-sample ${DATA}/sample_kmer_count \
    --o-chrom-lib ${DATA}/chrom_kmer_lib

# Build density matrix
python -m kmerize matrix \
    --chrom-lib ${DATA}/chrom_kmer_lib \
    --sample-count ${DATA}/sample_kmer_count \
    --output ${DATA}/cen_kmer_density_mat
```

### 2. `kmer_filter` — Specificity-based k-mer filtering

Filters k-mers by centromere specificity and chromosome specificity, producing the final specific k-mer set used downstream.

- **`t2t`**: Generate whole-genome (T2T) per-chromosome k-mer counts from FASTA files.
- **`centro`**: Centromere-specific k-mer filtering (compare centromere vs. whole-genome counts).
- **`chrom`**: Chromosome-specific k-mer filtering via inter-chromosome MSB.
- **`filter`**: Merge centro + chrom results into the final specific k-mer set.

```bash
# Step 0: Generate T2T whole-genome k-mer counts
python -m kmer_filter t2t \
    --t2t-dir ${DATA}/RawData/rice_t2t_70 \
    --output ${DATA}/T2T_kmer_count \
    --k 27

# Step 1: Centromere-specific filtering
python -m kmer_filter centro \
    --cen ${DATA}/sample_kmer_count \
    --t2t ${DATA}/T2T_kmer_count \
    --chrom-lib ${DATA}/chrom_kmer_lib \
    --output ${DATA}/centro_specific_score \
    --score-threshold 1

# Step 2: Chromosome-specific filtering (MSB)
python -m kmer_filter chrom \
    --density ${DATA}/cen_kmer_density_mat \
    --output ${DATA}/kmer_interchrom_msb.csv

# Step 3: Merge into final specific k-mer set
python -m kmer_filter filter \
    --msb ${DATA}/kmer_interchrom_msb.csv \
    --centro-score ${DATA}/centro_specific_score \
    --output ${DATA}/specific_kmers \
    --msb-ratio 0
```

### 3. `cluster` — Iterative NMF clustering

Computes a sample-wise mean Euclidean distance matrix from the k-mer density matrix via iterative subsampling + NMF.

- **`nmf`**: Iterative subsampling NMF → consensus distance matrix.

```bash
python -m cluster nmf \
    --density-mat ${DATA}/cen_kmer_density_mat \
    --output ${DATA}/iter_cluster \
    --n-iters 600 \
    --n-jobs 20
```

### 4. `deconv` — Centromere haplotype deconvolution (JAX/Flax)

Deconvolves NGS k-mer counts into centromere haplotype compositions using a hierarchical clustering tree and adaptive k-mer selection (F-value based). Uses JAX for GPU-accelerated optimization.

- **`run`**: Run deconvolution on all samples × all chromosomes.
- **`run-one`**: Run deconvolution on a single sample × chromosome pair (for SLURM array jobs).

```bash
python -m deconv run \
    --distance-mat ${DATA}/iter_cluster \
    --specific-kmers ${DATA}/specific_kmers \
    --density-mat ${DATA}/cen_kmer_density_mat \
    --ngs-kmers ${DATA}/NGS_kmer_count \
    --ngs-depth ${DATA}/NGS_depth.csv \
    --output ${DATA}/deconv_res \
    --n-steps 4000 \
    --seed 422 \
    --top-pct 0.2 \
    --sparsity-weight 0.01 \
    --top-k 2 \
    --level-stride 4 \
    --lr 0.005
```

### 5. `monomer` — Circular monomer unit reconstruction

Reconstructs circular centromere monomer units from deconvolution results and NGS read support (minimap2 PAF).

- **`prep`**: Extract principal haplotype k-mer counts from deconvolution results.
- **`cycles`**: Detect circular monomer units via de Bruijn graph cycle enumeration + NGS-supported chimera filtering.

```bash
# Step 1: Deconv results -> per-sample k-mer counts
python -m monomer prep \
    --res-dir ${DATA}/deconv_res \
    --component-dir ${DATA}/cen_kmer_density_mat \
    --out-dir ${DATA}/pre_kmer_count \
    --res-level 17 \
    --principal-prop 0.95

# Step 2: Align NGS reads to k-mer fasta with minimap2
# (minimap2 -k must match the cycles --kmer-len, both default to 27)
while IFS=_ read -r chrom sample; do
    minimap2 -t 20 -k 27 -w 1 -m 1 -n 1 -P \
        ${DATA}/pre_kmer_count/${chrom}_${sample}.fa \
        ${DATA}/art_fq/${sample}_1.fq \
        > ${DATA}/ngs_verify/${chrom}_${sample}_1.paf
    minimap2 -t 20 -k 27 -w 1 -m 1 -n 1 -P \
        ${DATA}/pre_kmer_count/${chrom}_${sample}.fa \
        ${DATA}/art_fq/${sample}_2.fq \
        > ${DATA}/ngs_verify/${chrom}_${sample}_2.paf
done < ${DATA}/pre_kmer_count/sample.list

# Step 3: Detect circular monomer units
python -m monomer cycles \
    --kmer-count-dir ${DATA}/pre_kmer_count \
    --ngs-verify-dir ${DATA}/ngs_verify \
    --out-dir ${DATA}/monomer_res \
    --align-threshold 0.8
```

## Installation

### Prerequisites

- Python >= 3.10
- `seqkit` (for `kmerize/extract_hor.sh`)
- `minimap2` (for `monomer` alignment step)
- GPU + CUDA (recommended for `deconv` module; falls back to CPU)

### Setup

```bash
git clone https://github.com/<your-username>/CenHaps.git
cd CenHaps
pip install -r requirements.txt
```

## Usage

Each module is run as a Python package (`python -m <module> <subcommand>`). The project root must be on `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

SLURM batch scripts are provided for each module (`run_*.sh`). Edit the `CODES` and `DATA` path variables at the top of each script to match your environment before submitting:

```bash
# Edit paths, then submit
sbatch cluster/run_cluster.sh
```

## Project Structure

```
CenHaps/
├── kmerize/          # HOR array k-mer extraction + density matrix
├── kmer_filter/      # Specificity-based k-mer filtering (centro + chrom + t2t)
├── cluster/          # Iterative NMF clustering
├── deconv/           # Haplotype deconvolution (JAX/Flax)
├── monomer/          # Circular monomer unit reconstruction
├── requirements.txt
├── LICENSE
└── README.md
```

## Data Conventions

All modules use fixed filename conventions — only directories are parameterized:

| File | Naming | Columns |
|------|--------|---------|
| HOR FASTA | `Chr{XX}_{sample}.fasta(.gz)` | — |
| Sample k-mer counts | `Chr{XX}_{sample}.csv` | `kmer, count, density` |
| Chromosome k-mer library | `Chr{XX}.csv` | `kmer` |
| Density matrix | `Chr{XX}.csv` | `kmer, <sample1>, <sample2>, ...` |
| Distance matrix | `Chr{XX}.csv` | consensus distance |
| Specific k-mers | `Chr{XX}.csv` | `kmer` |
| NGS k-mer counts | `{sample}.csv` | `kmer, count` |
| Deconv tree | `Chr{XX}_{sample}_tree.csv` | sample × group per cut level |
| Deconv props | `Chr{XX}_{sample}_props.csv` | long-format proportions |

## License

[MIT](LICENSE)
