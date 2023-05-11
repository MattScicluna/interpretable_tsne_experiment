'''
The complete MNIST experiment
Note that we vary 
    * the method of corruption ("permute" "remove" or "mean")
    * how we compute the gradient (w.r.t. gradient norm or the kl obj loss func)

This code utilizes the array jobs feature of a SLURM workload manager.
i.e. this script is run 10 times in parallel (job index is $SLURM_ARRAY_TASK_ID)
If you are not using SLURM, you would need to modify this code.
'''

#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9

#84:00:00
echo 'running job id: $SLURM_JOB_ID'

source ./set_environment_variables_public.sh

# Activate virtual env
module load python/3.8
source $ENV_LOC/bin/activate
module load python/3.8  # fixes problem with matplotlib

echo 'Running grad norm experiment for seed '$SLURM_ARRAY_TASK_ID ;
bash mnist_experiment_each_seed.sh "permute" "grad_norm" $SLURM_ARRAY_TASK_ID ;
bash mnist_experiment_each_seed.sh "mean" "grad_norm" $SLURM_ARRAY_TASK_ID ;

echo 'Running KL OBJ experiment for seed '$SLURM_ARRAY_TASK_ID ;
bash mnist_experiment_each_seed.sh "permute" "kl_obj" $SLURM_ARRAY_TASK_ID ;
bash mnist_experiment_each_seed.sh "mean" "kl_obj" $SLURM_ARRAY_TASK_ID ;
