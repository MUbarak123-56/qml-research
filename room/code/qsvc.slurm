#!/bin/bash
#SBATCH --mail-user=manda.li@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=10  
#SBATCH --ntasks=20
#SBATCH --time=05-00:00:00
#SBATCH --mem=150G
#SBATCH --output=qsvc.out
#SBATCH --error=qsvc.err
#SBATCH --job-name=parallel
#SBATCH --account=p_dsi
#SBATCH --partition=production

python qsvc.py