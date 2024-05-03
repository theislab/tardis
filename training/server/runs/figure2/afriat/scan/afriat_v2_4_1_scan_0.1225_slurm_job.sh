#!/bin/bash

#SBATCH -J tardis_run_afriat_scan
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 32
#SBATCH --mem=40G
#SBATCH --nice=0
#SBATCH -t 9:59:00

source activate tardis_env
jupyter nbconvert --execute --inplace --debug /home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure2/afriat/scan/afriat_v2_4_1_scan_0.1225.ipynb

