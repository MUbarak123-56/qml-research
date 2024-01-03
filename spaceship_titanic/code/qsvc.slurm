#!/bin/bash
#SBATCH --mail-user=mubarak.a.ganiyu@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1   
#SBATCH --ntasks=10
#SBATCH --time=05-00:00:00
#SBATCH --mem=300G
#SBATCH --output=qsvc.out
#SBATCH --error=qsvc.err
#SBATCH --job-name=parallel
#SBATCH --account=p_dsi
#SBATCH --partition=production

python qsvc.py