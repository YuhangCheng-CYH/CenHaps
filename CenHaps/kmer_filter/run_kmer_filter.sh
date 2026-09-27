#!/bin/bash
#SBATCH --job-name=kmer_filter
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200g
#SBATCH --time=900:00:00
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate py310

# ===========================================================================
# kmer_filter pipeline: specificity-based k-mer filtering (4 steps).
#   Step 0 t2t   : whole-genome FASTA -> per-chromosome k-mer counts
#   Step 1 centro : centromere-specific filtering (score >= threshold)
#   Step 2 chrom  : chromosome-specific filtering (inter-chrom MSB)
#   Step 3 filter : merge centro + chrom -> final specific k-mer set
# All input/output filenames are fixed by convention; only dirs are set below.
# ===========================================================================

# ---- Set these paths to match your environment ----
CODES=/public/home/Xlj20250201/CenHap/codes_20260709
DATA=/public/home/Xlj20250201/CenHap/00_data
export PYTHONPATH="${CODES}:${PYTHONPATH}"

# ---- Step 0: generate whole-genome (T2T) k-mer counts ----
python -m kmer_filter t2t \
    --t2t-dir "${DATA}/RawData/rice_t2t_70" \
    --output "${DATA}/T2T_kmer_count" \
    --k 27

# ---- Step 1: centromere-specific k-mer filtering ----
python -m kmer_filter centro \
    --cen "${DATA}/sample_kmer_count" \
    --t2t "${DATA}/T2T_kmer_count" \
    --chrom-lib "${DATA}/chrom_kmer_lib" \
    --output "${DATA}/centro_specific_score" \
    --score-threshold 1

# ---- Step 2: chromosome-specific k-mer filtering (MSB) ----
python -m kmer_filter chrom \
    --density "${DATA}/cen_kmer_density_mat" \
    --output "${DATA}/kmer_interchrom_msb.csv"

# ---- Step 3: merge into final specific k-mer set ----
python -m kmer_filter filter \
    --msb "${DATA}/kmer_interchrom_msb.csv" \
    --centro-score "${DATA}/centro_specific_score" \
    --output "${DATA}/specific_kmers" \
    --msb-ratio 0
