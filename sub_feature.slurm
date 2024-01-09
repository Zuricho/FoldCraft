#!/bin/bash
#SBATCH --job-name=feature
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --output=task_file/%x_%j.out
#SBATCH --error=task_file/%x_%j.err

module purge
module load miniconda3
source activate alphafold

./run_alphafold.sh \
-d data \
-o output \
-p monomer_ptm \
-i input/mono_set1/GB98.fasta \
-t 2022-01-01 \
-f

