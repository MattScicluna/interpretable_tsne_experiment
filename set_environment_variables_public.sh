'''
All the paths are defined here.
We create the paths in case they dont exist.
'''

#!/bin/bash

# Paths
export ENV_LOC=/path/to/python/envirionment
export RESULTS_DIR=path/to/folder/containing/results


export TSNE_OUTPUT_DIR_SIM=/path/to/t-SNE/embeddings/for/sim/data
export DATA_DIR_SIM=/path/to/sim/data

export DATA_DIR_MNIST=/path/to/MNIST/data
export TSNE_OUTPUT_DIR_MNIST=/path/to/t-SNE/embeddings/for/MNIST/data

export DATA_DIR_20NG=/path/to/20NG/data
export TSNE_OUTPUT_DIR_SARS=/path/to/t-SNE/embeddings/for/20NG/data
export W2V_DIR=/path/to/word2vec/model

export DATA_DIR_SARS=/path/to/SARS-CoV-2/data
export TSNE_OUTPUT_DIR_SARS=/path/to/t-SNE/embeddings/for/SARS-CoV-2/data
export RAW_SARS_DSET=path/to/SARS-CoV-2/raw/data
export METADATA_PATH=/path/to/SARS-CoV-2/metadata/file

# Make directories if they dont exist!
mkdir -p $RESULTS_DIR

mkdir -p $TSNE_OUTPUT_DIR_SIM
mkdir -p $DATA_DIR_SIM

mkdir -p $DATA_DIR_MNIST
mkdir -p $TSNE_OUTPUT_DIR_MNIST

mkdir -p $DATA_DIR_20NG
mkdir -p $TSNE_OUTPUT_DIR_SARS

mkdir -p $DATA_DIR_SARS
mkdir -p $TSNE_OUTPUT_DIR_SARS
