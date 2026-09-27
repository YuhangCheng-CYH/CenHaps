#!/bin/bash
#SBATCH --job-name=cluster
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
# cluster pipeline: iterative NMF clustering.
#   Input:  cen_kmer_density_mat/Chr{XX}.csv   (density matrix)
#   Output: iter_cluster/Chr{XX}.csv          (mean euclidean distance matrix)
# All input/output filenames are fixed by convention; only dirs are set below.
# ===========================================================================

# ---- Set these paths to match your environment ----
CODES=/public/home/Xlj20250201/CenHap/codes_20260709
DATA=/public/home/Xlj20250201/CenHap/00_data
export PYTHONPATH="${CODES}:${PYTHONPATH}"

# ---- iterative NMF distance matrix (parallel, default 600 iterations) ----
# --n-jobs 20: match SBATCH --cpus-per-task=20 (avoid using node-wide cores)
python -m cluster nmf \
    --density-mat "${DATA}/cen_kmer_density_mat" \
    --output "${DATA}/iter_cluster" \
    --n-jobs 20
