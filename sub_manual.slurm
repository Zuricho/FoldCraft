#!/bin/bash
#SBATCH --job-name=GA98_temp_1
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1
#SBATCH --output=task_file/%x_%j.out
#SBATCH --error=task_file/%x_%j.err

module purge
module load miniconda3
source activate parafold


# create manual feature
python parafold/create_manual_template.py \
--fasta_paths input/benchmark/GA98/GA98_temp_1.fasta \
--output_dir output \
--template_path template/GA98_temp_1.pdb

# run alphafold
./run_alphafold.sh \
-d data \
-o output \
-p monomer_ptm \
-u 0 \
-i \
input/benchmark/GA98/GA98_temp_1.fasta \
-t 2020-01-01 \
-r none \
-s     # skip msa

