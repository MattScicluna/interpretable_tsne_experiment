import os
import sys
from pathlib import Path
import pickle
import copy
import argparse
import json

import numpy as np

import skfeature
from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import fisher_score
from skfeature.utility import construct_W
from interpretable_tsne.tsne import TSNE

from attribution_experiment_individual_feats import remove_all_steps


def main(args):
    remove_all_steps(args, level='global')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Attribution experiment with Baselines')
    parser.add_argument('--tsne_output_dir', type=str,
                        help='directory of where t-SNE outputs are')
    parser.add_argument('--data_dir', type=str,
                        help='directory of where proceesed datasets are')
    parser.add_argument('--data_id', type=int, default=0,
                        help='id of processing')
    parser.add_argument('--run_id', type=int, default=0,
                        help='run id (random seed set as this for replication experiment)')
    parser.add_argument('--step', type=int, default=250,
                        help='Which step of attr to use in computations')
    parser.add_argument('--method', type=str,
                        help='What method of corruption to use: `permute`, `set_to_0`, `mean`, `remove`')
    parser.add_argument('--grad_style', type=str,
                        help='What gradient style to use: `grad_norm`, `kl_obj`, `mean_grad_norm`, `kl_obj_mean`')
    parser.add_argument('--indices_list', nargs='+', type=str,
                        help='What methods of indices computation to use (does random by default): `top_pc`, `top_fs`, `top_ls`, `top_ls_p_matrix`, `top_ls_q_matrix`, `top_attr_times_feat_unif`, `top_attr_unif`, `top_feat`')
    parser.add_argument('--final_csv_path', type=str,
                        help='where to save the final csv file')
    parser.add_argument('--remove_until', type=int, default=10,
                        help='Remove until this number of features')
    args = parser.parse_args()
    main(args)
