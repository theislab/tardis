#!/bin/bash

#SBATCH -J tardis_scib
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH --nice=0
#SBATCH -t 2-23:50:00

source activate tardis_env
python /home/icb/kemal.inecik/work/codes/tardis/notebooks/figure2c/calculate_scib.py /lustre/groups/ml01/workspace/kemal.inecik/tardis_data/_temporary/latent/suo_v01_01_2_tardis_15_latent.h5ad tardis batchcorrection integration_donor.integration_library_platform_coarse.organ.unreserved 0,1,2,3,4,5,6,7.8,9,10,11,12,13,14,15.16,17,18,19,20,21,22,23.24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 /lustre/groups/ml01/workspace/kemal.inecik/tardis_data/processed/dataset_complete_Suo.h5ad
