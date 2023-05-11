"""
This code creates all the datasets used in this work.
Note that we generate the 20 NewsGroups datasets, but we have not included results from that dataset (yet).
"""

#!/bin/bash

# Paths
source ./set_environment_variables_public.sh

# Activate virtual env
module load python/3.8 # if you are running this on compute canada
source $ENV_LOC/bin/activate

echo 'creating simulated datasets ...'
python src/set_up_sim_data.py --data_dir $DATA_DIR_SIM
echo 'finished'

echo 'creating mnist dataset ...'
python src/set_up_mnist.py --id 0 --size 10000 --data_dir $DATA_DIR_MNIST --num_pca 50
echo 'finished'

# Download GoogleNews-vectors-negative300.bin from the Google drive linked in the README, and place in $MODEL_DIR!
echo 'creating 20 ng dataset ...'
python src/set_up_20ng_word2vec.py --id 0 --size 12000 --data_dir $DATA_DIR_20NG --num_pca 0 --model_dir $W2V_DIR  # note we dont use PCA since Word2Vec already reduces dimension to 300!
echo 'finished'

# need to copy over the .mat and metadata files
echo 'creating SARS-CoV-2 datasets ...'

SARS_DSETS=('mutlist01_del_over0' 'mutlist01_del_withRef_over1' 'mutlist01_nodel_withRef_over0' 'mutlist01_del_over1' 'mutlist01_nodel_over0' 'mutlist01_nodel_withRef_over1' 'mutlist01_del_withRef_over0'   'mutlist01_nodel_over1')

for SARS_DSET in ${SARS_DSETS[*]}
do
    echo 'creating '$SARS_DSET
    python src/set_up_sars_cov_2.py --id 0  --data_dir $DATA_DIR_SARS/$SARS_DSET \
                                    --raw_data_path $RAW_SARS_DSET \
                                    --raw_data_name $SARS_DSET \
                                    --num_pca 100 \
                                    --metadata_path $METADATA_PATH
done
echo 'finished'
