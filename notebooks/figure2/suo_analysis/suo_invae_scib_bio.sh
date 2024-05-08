#!/bin/bash

#SBATCH -J suo_invae_scib
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 32
#SBATCH --mem=300G
#SBATCH --nice=0
#SBATCH -t 1-23:59:00

source activate tardis_env
jupyter nbconvert --execute --inplace --debug /home/icb/kemal.inecik/work/codes/tardis/notebooks/figure2/suo_analysis/analysis_suo_invae_bio.ipynb

