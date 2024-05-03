#!/bin/bash

#SBATCH -J tardis_latent_run
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=159G
#SBATCH --nice=0
#SBATCH -t 1-23:50:00

source activate tardis_env
python /home/icb/kemal.inecik/work/codes/tardis/notebooks/figure2c/collect_latent.py /lustre/groups/ml01/workspace/kemal.inecik/tardis_data/processed/dataset_complete_Suo.h5ad /lustre/groups/ml01/workspace/kemal.inecik/tardis_data/models/suo_v01_01_2_encode /lustre/groups/ml01/workspace/kemal.inecik/tardis_data/_temporary/latent/suo_v01_01_2_encode_tardis_15_latent.h5ad 15
