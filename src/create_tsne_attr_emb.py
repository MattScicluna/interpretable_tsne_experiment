import os
#import sys
from pathlib import Path
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from attribution_experiment_individual_feats import get_tsne_attributions 


def main():
    parser = argparse.ArgumentParser(description='Create t-SNE embeddings and attributions')
    parser.add_argument('--seed_value', type=int, default=0,
                        help='random seed (for subsetting, etc...)')
    parser.add_argument('--data_dir', type=str,
                        help='data loaded from {data_dir}/{data_id}/processed_data.npz')
    parser.add_argument('--tsne_output_dir', type=str,
                        help='embeddings/attributions are saved to {tsne_output_dir}/{run_id}/tsne_results_style={grad_style}.npz')
    parser.add_argument('--run_id', type=str,
                        help='results are saved to {results_dir}/{run_id}/tsne_results_style={grad_style}.npz')
    parser.add_argument('--data_id', type=int, default=0,
                        help='data loaded from {data_dir}/{data_id}/processed_data.npz')
    parser.add_argument('--grad_style', type=str,
                        help='What gradient style to use: `grad_norm`, `kl_obj`, `mean_grad_norm`, `kl_obj_mean`, `none`')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--n_iter', type=int, default=1000,
                        help='t-SNE n-iter parameter')
    parser.add_argument('--early_exaggeration', type=int, default=4,
                        help='t-SNE early exaggeration parameter')
    parser.add_argument('--learning_rate', type=int, default=500,
                        help='t-SNE learning rate parameter')
    args = parser.parse_args()

    get_tsne_attributions(args.seed_value,
                          args.data_dir,
                          args.tsne_output_dir,
                          args.run_id,
                          args.data_id,
                          args.grad_style,
                          args.perplexity,
                          args.n_iter,
                          args.early_exaggeration,
                          args.learning_rate)


if __name__ == '__main__':
    main()
