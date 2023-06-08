'''
The complexity experiment. This consists of two parts:
    * increasing the number of features, while keeping the number of samples constant at 1000
    * increasing the number of samples, while keeping the number of features constant at 10

This code contains SBATCH args for SLURM workload manager.
If you are not using SLURM, you can remove that part.
'''

#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

echo running job id: $SLURM_JOB_ID

# Paths
source ./set_environment_variables_public.sh

# Compute Canada uses modules. See: https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en. Ignore if not using. 
module load python/3.8

# Activate virtual env
source $ENV_LOC/bin/activate
module load python/3.8  # fixes problem with matplotlib

python src/complexity_experiment.py
