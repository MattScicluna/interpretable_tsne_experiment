import os
import sys
from pathlib import Path
from pprint import pprint
import pickle
import time
import copy
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu

from interpretable_tsne.tsne import TSNE
import skfeature
from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import fisher_score
from skfeature.utility import construct_W


def plot_graphs(result_dir, method, grad_style, final_csv_path):

    if method == 'mean':
        global_method = 'remove'
    else:
        global_method = method

    # Class-level
    levels_index=['random',
                  'top_ls_class_p_matrix',
                  'top_ls_class_q_matrix',
                  'top_attr_unif_class',
                  'top_feat_class',
                  'top_attr_times_feat_unif_class']
    level_names=['Random',
                 'Top Laplace Score Per Class (using P)',
                 'Top Laplace Score Per Class (using Q)',
                 'Top Attribution Per Class',
                 'Top Feature Value Per Class',
                 'Top Attribution $\cdot$ Feature Value Per Class']
    rseeds = range(30, 40)
    _fname = '{}_feats_method={}_grad_style={}/attr_exp_results_{}_feats_full.csv'.format('class', method, grad_style, 'class')
    try:
        df = pd.read_csv(Path(final_csv_path) / _fname, index_col=0)
        print('Loaded data from {}. Skipping...'.format(_fname))
    except:
        print('Could not load data from {}. Creating...'.format(_fname))
        dfs = []
        for i in rseeds:
            _df = pd.read_csv(Path(result_dir) / '{}_feats_method={}_grad_style={}/attr_exp_results_{}_feats_{}.csv'.format('class', method, grad_style, 'class', i))
            _df['Grad Style'] = grad_style
            dfs.append(_df)
        df = pd.concat(dfs)
        df.drop(columns=['Unnamed: 0'])
        df.to_csv(Path(final_csv_path) / _fname)

    # Global
    _fname = '{}_feats_method={}_grad_style={}/attr_exp_results_{}_feats_full.csv'.format('global', global_method, grad_style, 'global')
    levels_index=['random',
                  'top_fs',
                  'top_pc',
                  'top_ls_p_matrix',
                  'top_ls_q_matrix',
                  'top_feat',
                  'top_attr_unif',
                  'top_attr_times_feat_unif']
    level_names=['Random',
                 'Top Fisher Score',
                 'Top PC',
                 'Top Laplace Score (Using P)',
                 'Top Laplace Score (Using Q)',
                 'Top Feature Value',
                 'Attribution',
                 'Attribution $\cdot$ Feature Value'] 
    rseeds = range(30, 40)
    try:
        df = pd.read_csv(Path(final_csv_path) / _fname, index_col=0)
        print('Loaded data from {}. Skipping...'.format(_fname))
    except:
        print('Could not load data from {}. Creating...'.format(_fname))
        dfs = []
        for i in rseeds:
            _df = pd.read_csv(Path(result_dir) / '{}_feats_method={}_grad_style={}/attr_exp_results_{}_feats_{}.csv'.format('global', global_method, grad_style, 'global', i))
            _df['Grad Style'] = grad_style
            dfs.append(_df)
        df = pd.concat(dfs)
        df.drop(columns=['Unnamed: 0'])
        df.to_csv(Path(final_csv_path) / _fname)

    # Individual
    _fname = '{}_feats_method={}_grad_style={}/attr_exp_results_{}_feats_full.csv'.format('indv', method, grad_style, 'individual')
    levels_index=['random',
                  'top_attr_ge_0',
                  'top_attr',
                  'feat_size',
                  'attr_feat']
    level_names=['Random',
                 'Attribution $>0$',
                 'Attribution',
                 'Feature Value',
                 'Attribution $\cdot$ Feature Value']
    rseeds = range(20, 30)
    try:
        df = pd.read_csv(Path(final_csv_path) / _fname, index_col=0)
        print('Loaded data from {}. Skipping...'.format(_fname))
    except:
        print('Could not load data from {}. Creating...'.format(_fname))
        dfs = []
        for i in rseeds:
            _df = pd.read_csv(Path(result_dir) / '{}_feats_method={}_grad_style={}/attr_exp_results_{}_feats_{}.csv'.format('indv', method, grad_style, 'individual', i))
            _df['Grad Style'] = grad_style
            dfs.append(_df)
        df = pd.concat(dfs)
        df.drop(columns=['Unnamed: 0'])
        df.to_csv(Path(final_csv_path) / _fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Attribution experiment with Baselines')
    parser.add_argument('--result_dir', type=str,
                        help='directory of where per-experiment result csvs are')
    parser.add_argument('--method', type=str,
                        help='Which method of corruption to use: `mean`, `permute`')
    parser.add_argument('--grad_style', type=str,
                        help='What gradient style to use: `grad_norm`, `kl_obj`, `mean_grad_norm`, `kl_obj_mean`, `none`')
    parser.add_argument('--final_csv_path', type=str,
                        help='Where to put the output file')
    args = parser.parse_args()
    plot_graphs(args.result_dir, args.method, args.grad_style, args.final_csv_path)
