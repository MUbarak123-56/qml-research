#!/bin/bash
#SBATCH --mail-user=mubarak.a.ganiyu@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1   
#SBATCH --ntasks=10
#SBATCH --time=03-00:00:00
#SBATCH --mem=70G
#SBATCH --output=vqc_pauli_best.out
#SBATCH --error=vqc_pauli_best.err
#SBATCH --job-name=parallel
#SBATCH --account=p_dsi
#SBATCH --partition=production

python vqc_pauli_best.py