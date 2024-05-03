#!/bin/bash

#SBATCH -J tardis_latent_run2
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=99G
#SBATCH --nice=0
#SBATCH -t 1-23:50:00

source activate tardis_env
python /home/icb/kemal.inecik/work/codes/tardis/notebooks/figure2c/append_sublatents.py /lustre/groups/ml01/workspace/kemal.inecik/tardis_data/_temporary/latent/suo_v01_02_2_tardis_30_latent.h5ad organ.unreserved 0,1,2,3,4,5,6,7.8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 30
