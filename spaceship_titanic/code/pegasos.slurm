#!/bin/bash
#SBATCH --mail-user=mubarak.a.ganiyu@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1   
#SBATCH --ntasks=10
#SBATCH --time=07-00:00:00
#SBATCH --mem=70G
#SBATCH --output=pegasos.out
#SBATCH --error=pegasos.err
#SBATCH --job-name=parallel
#SBATCH --account=p_dsi
#SBATCH --partition=production

python pegasos.py