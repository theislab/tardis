#!/bin/bash

#SBATCH -J tardis_run_afriat_scan
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=159G
#SBATCH --nice=0
#SBATCH -t 1-23:50:00

source activate tardis_env
jupyter nbconvert --execute --inplace --debug /home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure2/afriat/scan/afriat_v2_4_1_scan_0.20.ipynb

