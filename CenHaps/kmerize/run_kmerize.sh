#!/bin/bash
#SBATCH --job-name=kmerize
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
# Statistical analysis of k-mers from the hor array on the centromere.
#   1. extract : bed + fasta  -> cen155/Chr{XX}_{samplename}.fasta.gz  (shell)
#   2. kmer    : cen155       -> sample_kmer_count/ + chrom_kmer_lib/   (python)
#   3. matrix  : lib + counts -> cen_kmer_density_mat/Chr{XX}.csv     (python)
# All input/output filenames are fixed by convention; only dirs are set below.
# ===========================================================================

# ---- Set these paths to match your environment ----
CODES=/public/home/Xlj20250201/CenHap/codes_20260709
DATA=/public/home/Xlj20250201/CenHap/00_data
export PYTHONPATH="${CODES}:${PYTHONPATH}"

# ---- 1a: extract cen155 (pure shell, seqkit) ----
bash "${CODES}/kmerize/extract_hor.sh"

# ---- 1b: generate sample k-mer counts + chromosome k-mer libs ----
python -m kmerize kmer \
    --k 27 \
    --hor "${DATA}/hor" \
    --o-sample "${DATA}/sample_kmer_count" \
    --o-chrom-lib "${DATA}/chrom_kmer_lib"

# ---- 1c: build density matrix per chromosome ----
python -m kmerize matrix \
    --chrom-lib "${DATA}/chrom_kmer_lib" \
    --sample-count "${DATA}/sample_kmer_count" \
    --output "${DATA}/cen_kmer_density_mat"
