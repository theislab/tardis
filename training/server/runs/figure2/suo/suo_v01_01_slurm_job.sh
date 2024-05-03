#!/bin/bash

#SBATCH -J tardis_run
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=159G
#SBATCH --nice=0
#SBATCH -t 1-23:50:00

source activate tardis_env
python /home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure2/suo/suo_v01_01.py

