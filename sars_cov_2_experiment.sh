'''
Computes the embeddings for the SARS-CoV-2 Experiments
Note that we only use embeddings of "mutlist01_del_withRef_over1" and "mutlist01_del_over1"

Naming convention:
del = contains deletions (not just mutations)
withRef = contains reference column (implicitly encodes missingness data
over1 = only contains mutations that occur more then 1 time
'''

#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=30G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#84:00:00
echo running job id: $SLURM_JOB_ID

source ./set_environment_variables_public.sh

GRAD_STYLE="grad_norm"
SEED=0

# Activate virtual env
source $ENV_LOC/bin/activate
module load python/3.8

echo 'creating SARS-CoV-2 embeddings...'

SARS_DSETS=('mutlist01_del_over0' 'mutlist01_del_withRef_over1' 'mutlist01_nodel_withRef_over0' 'mutlist01_del_over1' 'mutlist01_nodel_over0' 'mutlist01_nodel_withRef_over1' 'mutlist01_del_withRef_over0'   'mutlist01_nodel_over1')

for SARS_DSET in ${SARS_DSETS[*]}
do
    echo 'creating '$SARS_DSET ' embeddings'
    python src/create_tsne_attr_emb.py --seed_value $SEED \
    --data_dir $DATA_DIR_SARS/$SARS_DSET \
    --tsne_output_dir $TSNE_OUTPUT_DIR_SARS/$SARS_DSET \
    --run_id $SEED \
    --data_id 0 \
    --grad_style $GRAD_STYLE
    echo 'creating '$SARS_DSET ' figures'
done
echo 'finished'
