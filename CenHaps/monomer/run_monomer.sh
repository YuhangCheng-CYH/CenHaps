#!/bin/bash
#SBATCH --job-name=monomer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200g
#SBATCH --time=900:00:00
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate py310

# ===========================================================================
# monomer pipeline: circular centromere monomer unit reconstruction.
#   Step 1 (prep):   deconv_res/Chr{XX}_{sample}_props.csv + _tree.csv
#                    + cen_kmer_density_mat/Chr{XX}.csv
#                    -> pre_kmer_count/Chr{XX}_{sample}.csv / .fa / sample.list
#   Step 2 (align):  minimap2 reads vs k-mer fasta (art_fq/{sample}_1.fq / _2.fq)
#                    -> ngs_verify/Chr{XX}_{sample}_1.paf / _2.paf
#   Step 3 (cycles): k-mer counts + PAF read support
#                    -> monomer_res/Chr{XX}_{sample}_cycles_summary.tsv
#                       monomer_res/Chr{XX}_{sample}_cycles_detail.json
# All input/output filenames are fixed by convention; only dirs are set below.
# NOTE: minimap2 -k must match the cycles --kmer-len (both default to 27).
# ===========================================================================

# ---- Set these paths to match your environment ----
CODES=/public/home/Xlj20250201/CenHap/codes_20260310
DATA=/public/home/Xlj20250201/CenHap/00_data
export PYTHONPATH="${CODES}:${PYTHONPATH}"

# Step 1: deconv results -> per-sample k-mer counts
python -m monomer prep \
    --res-dir "${DATA}/deconv_res" \
    --component-dir "${DATA}/cen_kmer_density_mat" \
    --out-dir "${DATA}/pre_kmer_count" \
    --res-level 17 \
    --principal-prop 0.95

# Step 2: align NGS reads to the k-mer fasta with minimap2
mkdir -p "${DATA}/ngs_verify"

while IFS=_ read -r chrom sample; do

    minimap2 -t 20 -k 27 -w 1 -m 1 -n 1 -P "${DATA}/pre_kmer_count/${chrom}_${sample}.fa" \
        "${DATA}/art_fq/${sample}_1.fq" > "${DATA}/ngs_verify/${chrom}_${sample}_1.paf"

    minimap2 -t 20 -k 27 -w 1 -m 1 -n 1 -P "${DATA}/pre_kmer_count/${chrom}_${sample}.fa" \
        "${DATA}/art_fq/${sample}_2.fq" > "${DATA}/ngs_verify/${chrom}_${sample}_2.paf"

done < "${DATA}/pre_kmer_count/sample.list"

# Step 3: detect circular monomer units
python -m monomer cycles \
    --kmer-count-dir "${DATA}/pre_kmer_count" \
    --ngs-verify-dir "${DATA}/ngs_verify" \
    --out-dir "${DATA}/monomer_res" \
    --align-threshold 0.8
