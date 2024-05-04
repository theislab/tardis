#!/bin/bash

#SBATCH -J af_ablation_figr
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 32
#SBATCH --mem=300G
#SBATCH --nice=0
#SBATCH -t 3-00:00:00

source activate tardis_env
jupyter nbconvert --execute --inplace --debug /home/icb/kemal.inecik/work/codes/tardis/notebooks/ablation_analysis.ipynb
