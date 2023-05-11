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

from interpretable_tsne.tsne import TSNE
import skfeature
from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import fisher_score
from skfeature.utility import construct_W


def agg_attrs(attrs, num_keep_idx=7):
    top_indices = np.argsort(np.abs(attrs), axis=1)[:,num_keep_idx:]
    counts = []
    for i in range(attrs.shape[1]):
        counts.append(np.count_nonzero(top_indices == i))
    return np.array(counts)/np.array(counts).sum()


def main(args):
    os.makedirs(args.final_csv_path, exist_ok=True)

    # Create dataframe
    df = pd.DataFrame({'Data Seed': [], 
                       't-SNE Random Seed': [], 
                       'Dataset Name': [], 
                       'Feat Averages': [], 
                       'Attr Averages': [], 
                       'Feat Averages Agg': [],
                       'Attr Averages Agg': [],
                       'Grad Style': [], 
                       'Step': []
                      })

    for data_seed in args.seeds:
        tsne_output_dir = '{}/{}'.format(args.tsne_output_dir, data_seed)
        for tsne_seed in args.seeds:
            # load attributions
            try:
                arr_obj = np.load(Path(tsne_output_dir) / str(tsne_seed)/ 'tsne_results_style={}.npz'.format(args.grad_style), allow_pickle=True)
            except FileNotFoundError:
                print('Did not find {}. Terminating'.format(tsne_output_dir))
                raise Exception
            data_obj = np.load('{}/{}/processed_data.npz'.format(args.data_dir, data_seed), 'rb')
            data, labels = data_obj['X_reduced'], data_obj['labels']

            #  Load array object from tsne_output_dir
            out = arr_obj['out'].item()
            attrs = out['attrs'][args.step] #[0]
            attrs = attrs[0]
            attrs[np.isnan(attrs)] = 0 # remove nans
            attrs[attrs > 1] = 1 # remove exploding values
            attrs[attrs < -1] = -1 # remove exploding values

            for _class in np.unique(labels):
                df = df.append({'Data Seed': data_seed,
                                't-SNE Random Seed': tsne_seed,
                                'Dataset Name': args.dset_name,
                                'Class': _class,
                                'Feat Averages': json.dumps(np.abs(data[labels == _class].mean(0)).tolist()),
                                'Attr Averages': json.dumps(np.abs(attrs[labels == _class].mean(0)).tolist()),
                                'Feat Averages Agg': json.dumps(agg_attrs(data[labels == _class]).tolist()),
                                'Attr Averages Agg': json.dumps(agg_attrs(attrs[labels == _class]).tolist()),
                                'Grad Style': args.grad_style,
                                'Step': args.step
                               },
                               ignore_index=True)

            df.to_csv(os.path.join(args.final_csv_path, 'sim_data_attr_exp.csv'))
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Attribution experiment with Baselines')
    parser.add_argument('--tsne_output_dir', type=str,
                        help='directory of where t-SNE objects are')
    parser.add_argument('--data_dir', type=str,
                        help='directory of where proceesed datasets are')
    parser.add_argument('--dset_name', type=str,
                        help='Name of simulated dset')
    parser.add_argument('--step', type=int, default=250,
                        help='Which step of attr to use in computations')
    parser.add_argument('--seeds', nargs='+', type=str,
                        help='dataset/t-SNE seeds used')
    parser.add_argument('--final_csv_path', type=str,
                        help='where to save the final csv file')
    parser.add_argument('--grad_style', type=str,
                        help='What gradient style to use: `grad_norm`, `kl_obj`, `mean_grad_norm`, `kl_obj_mean`, `none`')
    args = parser.parse_args()
    main(args)
