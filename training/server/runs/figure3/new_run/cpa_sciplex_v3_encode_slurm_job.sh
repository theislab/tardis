#!/bin/bash

#SBATCH -J sciplex_few_unreserved
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 32
#SBATCH --mem=40G
#SBATCH --nice=0
#SBATCH -t 9:59:00

source activate tardis_env
jupyter nbconvert --execute --inplace --debug /home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure3/new_run/cpa_sciplex_v3_encode.ipynb

