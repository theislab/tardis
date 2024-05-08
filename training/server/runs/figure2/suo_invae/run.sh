#!/bin/bash

# Path to the Python files
DIRECTORY="/home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure2/suo_invae"

# Activate the Python environment
source activate invae_env

# Loop over all Python files in the directory
for file in $DIRECTORY/*.py
do
    # Extract the basename without the extension
    BASENAME=$(basename "$file" .py)

    # Create a SLURM script for each Python file with a specific name
    SCRIPT_NAME="$DIRECTORY/${BASENAME}_slurm_job.sh"
    cat << EOF > $SCRIPT_NAME
#!/bin/bash

#SBATCH -J invae_suo
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --nice=0
#SBATCH -t 1-23:59:00

source activate invae_env
python $file

EOF

    # Submit the job
    sbatch $SCRIPT_NAME

sleep 1
done
