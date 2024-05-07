#!/bin/bash

# Path to the Python files
DIRECTORY="/home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure2/suo_invae"

# Activate the Python environment
source activate cpa

# Loop over all Python files in the directory
for file in $DIRECTORY/*.py
do
    # Extract the basename without the extension
    BASENAME=$(basename "$file" .py)

    # Create a SLURM script for each Python file with a specific name
    SCRIPT_NAME="$DIRECTORY/${BASENAME}_slurm_job.sh"
    cat << EOF > $SCRIPT_NAME
#!/bin/bash

#SBATCH -J tardis_run
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=159G
#SBATCH --nice=0
#SBATCH -t 1-23:50:00

source activate cpa
python $file

EOF

    # Submit the job
    sbatch $SCRIPT_NAME

sleep 1
done
