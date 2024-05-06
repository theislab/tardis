#!/bin/bash

# Path to the Python files
DIRECTORY="/home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure3"

# Activate the Python environment
source activate tardis_env

# Loop over all Python files in the directory
for file in $DIRECTORY/*.ipynb
do
    # Extract the basename without the extension
    BASENAME=$(basename "$file" .ipynb)

    # Define the name for the SLURM script
    SCRIPT_NAME="$DIRECTORY/${BASENAME}_slurm_job.sh"

    # Check if the SLURM script already exists
    if [ -f "$SCRIPT_NAME" ]; then
        echo "Script file $SCRIPT_NAME already exists, skipping creation and submission."
        continue
    fi

    # Create a SLURM script for each Python file with a specific name
    cat << EOF > $SCRIPT_NAME
#!/bin/bash

#SBATCH -J tardis_braun
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=159G
#SBATCH --nice=0
#SBATCH -t 1-23:50:00

source activate tardis_env
jupyter nbconvert --execute --inplace --debug $file

EOF

    # Submit the job
    sbatch $SCRIPT_NAME
done
