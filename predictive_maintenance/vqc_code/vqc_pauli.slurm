#!/bin/bash
#SBATCH --mail-user=mubarak.a.ganiyu@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=10   
#SBATCH --ntasks=24
#SBATCH --time=07-00:00:00
#SBATCH --mem=150G
#SBATCH --output=vqc_pauli.out
#SBATCH --error=vqc_pauli.err
#SBATCH --job-name=parallel
#SBATCH --account=p_dsi
#SBATCH --partition=production

python vqc_pauli.py