#!/bin/bash
#SBATCH --job-name=extract_cen155
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
# Step 1a of the kmerize pipeline: slice centromere sequences with seqkit.
#   Input:  {BED_DIR}/Chr{XX}_{samplename}.bed
#           {FASTA_DIR}/{samplename}.fasta(.gz)
#           {SAMPLE_LIST}   (one sample name per line)
#   Output: {OUT_DIR}/Chr{XX}_{samplename}.fasta.gz
# Filenames are fixed by convention; only dirs are set below.
# ===========================================================================

# ---- Set these paths to match your environment ----
BED_DIR=/public/home/Xlj20250201/CenHap/00_data/cen_bed
FASTA_DIR=/public/home/Xlj20250201/CenHap/00_data/RawData/rice_t2t_70
SAMPLE_LIST=/public/home/Xlj20250201/CenHap/00_data/RawData/all_70.list
OUT_DIR=/public/home/Xlj20250201/CenHap/00_data/hor

mkdir -p "${OUT_DIR}"

while IFS= read -r samplename; do
	for i in $(seq -f "%02g" 1 12);do
		if [ -f "${BED_DIR}/Chr${i}_${samplename}.bed" ]; then
			if [ -f "${FASTA_DIR}/${samplename}.fasta.gz" ]; then
				seqkit subseq --bed "${BED_DIR}/Chr${i}_${samplename}.bed" "${FASTA_DIR}/${samplename}.fasta.gz" -o "${OUT_DIR}/Chr${i}_${samplename}.fasta.gz"
			elif [ -f "${FASTA_DIR}/${samplename}.fasta" ]; then
				seqkit subseq --bed "${BED_DIR}/Chr${i}_${samplename}.bed" "${FASTA_DIR}/${samplename}.fasta" -o "${OUT_DIR}/Chr${i}_${samplename}.fasta.gz"
			fi
		fi
	done
done < "${SAMPLE_LIST}"
