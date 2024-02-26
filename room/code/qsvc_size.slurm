#!/bin/bash
#SBATCH --mail-user=manda.li@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1   
#SBATCH --ntasks=10
#SBATCH --time=10-00:00:00
#SBATCH --mem=100G
#SBATCH --output=qsvc_size.out
#SBATCH --error=qsvc_size.err
#SBATCH --job-name=parallel
#SBATCH --account=p_dsi
#SBATCH --partition=production

python qsvc_size.py