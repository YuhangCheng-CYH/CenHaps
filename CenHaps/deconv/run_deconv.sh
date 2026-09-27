#!/bin/bash
#SBATCH --job-name=deconv
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

export LD_LIBRARY_PATH=/public/home/Xlj20250201/miniconda3/envs/py310/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/public/home/Xlj20250201/miniconda3/envs/py310

srun nvidia-smi

# ===========================================================================
# deconv pipeline: centromere haplotype deconvolution.
#   Input:  iter_cluster_new/Chr{XX}.csv       (distance matrix)
#           specific_kmers/Chr{XX}.csv          (specific kmer set)
#           cen_kmer_density_mat/Chr{XX}.csv  (density matrix)
#           NGS_kmers/{sample}.tsv               (NGS kmer counts)
#           NGS_depth/{sample}.chr.stat.gz       (NGS per-chrom depth)
#   Output: deconv_res/Chr{XX}_{sample}_tree.csv   (clustering groups)
#           deconv_res/Chr{XX}_{sample}_props.csv  (haplotype proportions)
# All input/output filenames are fixed by convention; only dirs are set below.
# ===========================================================================

# ---- Set these paths to match your environment ----
CODES=/public/home/Xlj20250201/CenHap/codes_20260709
DATA=/public/home/Xlj20250201/CenHap/00_data
export PYTHONPATH="${CODES}:${PYTHONPATH}"

python -m deconv run \
    --distance-mat "${DATA}/iter_cluster" \
    --specific-kmers "${DATA}/specific_kmers" \
    --density-mat "${DATA}/cen_kmer_density_mat" \
    --ngs-kmers "${DATA}/NGS_kmer_count" \
    --ngs-depth "${DATA}/NGS_depth.csv" \
    --output "${DATA}/deconv_res" \
    --n-steps 4000 \
    --seed 422 \
    --top-pct 0.2 \
    --sparsity-weight 0.01 \
    --top-k 2 \
    --level-stride 4 \
    --lr 0.005


