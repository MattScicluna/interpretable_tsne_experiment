'''
This code creates all the t-SNE embeddings.
It loops over random dataset (DATA_SEED) and t-SNE random initialization (TSNE_SEED).
'''

#!/bin/bash

DSET_NAME=$1 # dataset name
TSNE_OUTPUT_DIR=$2 # where to save embeddings
DATA_DIR=$3 # path to dataset
GRAD_STYLE=$4 # what kind of gradient to use: 'gram_norm' or 'kl_obj'

echo "Performing experiment on "$DSET_NAME

# we are varying both the t-SNE random seed and the data random seed at the same time
for DATA_SEED in {0..9}
    do
    for TSNE_SEED in {0..9}
    do
        python src/create_tsne_attr_emb.py \
        --seed_value $TSNE_SEED \
        --data_dir $DATA_DIR/$DSET_NAME \
        --tsne_output_dir $TSNE_OUTPUT_DIR/$DSET_NAME/$DATA_SEED \
        --run_id $TSNE_SEED \
        --data_id $DATA_SEED \
        --grad_style $GRAD_STYLE &
    done
done

wait

echo 'finished'
