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


def make_df(file_list, results_dir):
    full_df = pd.DataFrame()
    for file in file_list:
        data = pd.read_csv(os.path.join(results_dir, file, 'sim_data_attr_exp.csv'))
        full_df = pd.concat([full_df, data])
    full_df = full_df.reset_index()
    full_df = full_df.drop(columns=['Unnamed: 0', 'index'])
    full_df['Class'] = pd.Categorical(full_df['Class'])
    return full_df


def make_columns_from_string_list(full_df, colname='Feat Averages'):
    _means = pd.DataFrame(full_df[colname].str.split(',', expand=True).values, 
                          columns = ['{} {}'.format(colname, i) for i in range(10)])
    _means['{} 0'.format(colname)] = _means['{} 0'.format(colname)].str[1:]
    _means['{} 9'.format(colname)] = _means['{} 9'.format(colname)].str[:-1]
    _means = _means.astype(float)
    full_df = full_df.drop(columns=[colname])
    full_df[_means.columns.tolist()] = _means
    return full_df


def get_dset_info_from_desc(string):
    toks = string.split('_')
    num_clusters =  toks[1]
    eff1 = toks[3].split('=')[1]
    if num_clusters == '2':
        eff2 = 'NA'
    else:
        eff2 = toks[4].split('=')[1]
    return num_clusters, eff1, eff2


def _get_permutation_score(data, labels, n_permutations=10000):
    #assert 1 in np.unique(labels)
    #assert 0 in np.unique(labels)
    #assert len(np.unique(labels)) == 2
    statistic = data[labels==1].mean()
    C = 0
    for _ in range(n_permutations):
        new_labels = np.random.permutation(labels)
        new_statistic = data[new_labels==1].mean()
        if new_statistic >= statistic:
            C += 1
    return (C + 1) / (n_permutations + 1) #, statistic
    #Where C is the number of permutations whose score >= the true score.
    #The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.


def _get_mannwhitneyu_score(data, labels):
    return mannwhitneyu(data[labels==0], data[labels==1], use_continuity=True, alternative='less').pvalue


def get_permutation_score(colname, data):
    data = data[(data['variable'].isin(['{} {}'.format(colname, i) for i in range(10)]))]
    is_1 = data.apply(lambda row: row['variable'] == '{} 0'.format(colname), axis=1)
    is_2 = data.apply(lambda row: row['variable'] == '{} 1'.format(colname), axis=1)
    is_3 = data.apply(lambda row: row['variable'] == '{} 2'.format(colname), axis=1)
    cond1 = ((data['Class'] == 0.0) & (is_1 | is_2))
    cond2 = ((data['Class'] == 1.0) & (is_1 | is_2))
    cond3 = ((data['Class'] == 2.0) & (is_1 | is_3))
    cond4 = ((data['Class'] == 3.0) & (is_1 | is_3))

    pos_idx= (cond1 | cond2 | cond3 | cond4)
    neg_idx= ~(cond1 | cond2 | cond3 | cond4)

    labels = np.zeros(shape=cond1.shape[0])
    labels[pos_idx] = 1

    return (_get_mannwhitneyu_score(data['value'].values, labels), #_get_permutation_score(data['value'].values, labels),
            data['value'].values[labels==1].mean(),
            data['value'].values[labels==1].std(),
            data['value'].values[labels==0].mean(),
            data['value'].values[labels==0].std())


def main(args):
    full_df = make_df(args.dsets, args.results_dir)

    full_df = make_columns_from_string_list(full_df, 'Feat Averages')
    full_df = make_columns_from_string_list(full_df, 'Attr Averages')
    full_df = make_columns_from_string_list(full_df, 'Feat Averages Agg')
    full_df = make_columns_from_string_list(full_df, 'Attr Averages Agg')
    full_df[['Num Cluster', 'Effect 1', 'Effect 2']] = full_df.apply(lambda row : get_dset_info_from_desc(row['Dataset Name']), axis=1, result_type='expand')
    
    filter2 = (full_df['Effect 1'] == '2') & (full_df['Effect 2'].isin(['NA', '1']))
    filter3 = (full_df['Effect 1'] == '3') & (full_df['Effect 2'].isin(['NA', '1','2']))
    filter4 = (full_df['Effect 1'] == '4') & (full_df['Effect 2'].isin(['NA', '1', '2', '3']))
    filter6 = (full_df['Effect 1'] == '6') & (full_df['Effect 2'].isin(['NA', '1', '2', '3', '5']))
    #filter8 = (full_df['Effect 1'] == '8') & (full_df['Effect 2'].isin(['NA', '1', '2', '3', '5', '7']))

    # get rid of big effect sizes
    #full_df = full_df[full_df['Effect 1'].isin(['1', '2', '3', '4', '6']) & full_df['Effect 2'].isin(['NA', '1', '2', '3', '5'])]
    full_df = full_df[filter2 | filter3 | filter4 | filter6]

    #  Save this dataframe
    full_df.to_csv(os.path.join(args.final_csv_path, 'sim_data_attrs.csv'))

    # Now only looking at 4 cluster datasets
    df_4_cluster = full_df[full_df['Num Cluster'] == '4'].groupby(['Dataset Name', 'Class', 'Data Seed'])[['Attr Averages {}'.format(i) for i in range(10)]].mean()
    df_4_cluster['Effect 1'] = full_df[full_df['Num Cluster'] == '4'].groupby(['Dataset Name', 'Class', 'Data Seed'])['Effect 1'].first()
    df_4_cluster['Effect 2'] = full_df[full_df['Num Cluster'] == '4'].groupby(['Dataset Name', 'Class', 'Data Seed'])['Effect 2'].first()
    df_4_cluster = df_4_cluster.reset_index().melt(id_vars=['Dataset Name', 'Effect 1', 'Effect 2', 'Class', 'Data Seed'])
    df_4_cluster2 = full_df[full_df['Num Cluster'] == '4'].groupby(['Dataset Name', 'Class', 'Data Seed'])[['Feat Averages {}'.format(i) for i in range(10)]].mean()
    df_4_cluster2['Effect 1'] = full_df[full_df['Num Cluster'] == '4'].groupby(['Dataset Name', 'Class', 'Data Seed'])['Effect 1'].first()
    df_4_cluster2['Effect 2'] = full_df[full_df['Num Cluster'] == '4'].groupby(['Dataset Name', 'Class', 'Data Seed'])['Effect 2'].first()
    df_4_cluster2 = df_4_cluster2.reset_index().melt(id_vars=['Dataset Name', 'Effect 1', 'Effect 2', 'Class', 'Data Seed'])
    df_4_cluster = pd.concat([df_4_cluster, df_4_cluster2])
    datasets = np.unique(df_4_cluster['Dataset Name'])

    p_vals_df = pd.DataFrame({'Effect 1': [],
                          'Effect 2': [],
                          'class': [],
                          'mean of sig attrs': [],
                          'mean of non-sig attrs': [],
                          'std of sig attrs': [],
                          'std of non-sig attrs': [],
                          'attr p-val': [],
                          'mean of sig feats': [],
                          'mean of non-sig feats': [],
                          'std of sig feats': [],
                          'std of non-sig feats': [],
                          'feat p-val': []
                         })
    for dataset in datasets:
        for _class in [0.0, 1.0, 2.0, 3.0]:
            _class_label = str(int(_class + 1))
            _data = df_4_cluster[(df_4_cluster['Dataset Name'] == dataset) & (df_4_cluster['Class'].isin([_class]))]
            pval_aa, m13_aa, s13_aa, m49_aa, s49_aa = get_permutation_score('Attr Averages', _data)
            pval_fa, m13_fa, s13_fa, m49_fa, s49_fa = get_permutation_score('Feat Averages', _data)

            p_vals_df = p_vals_df.append({'Effect 1': _data['Effect 1'].iloc[0],
                                          'Effect 2': _data['Effect 2'].iloc[0],
                                          'class': _class_label,
                                          'mean of sig attrs': m13_aa,
                                          'mean of non-sig attrs': m49_aa,
                                          'std of sig attrs': s13_aa,
                                          'std of non-sig attrs': s49_aa,
                                          'attr p-val': pval_aa,
                                          'mean of sig feats': m13_fa,
                                          'mean of non-sig feats': m49_fa,
                                          'std of sig feats': s13_fa,
                                          'std of non-sig feats': s49_fa,
                                          'feat p-val': pval_fa}, ignore_index=True)
    p_vals_df.to_csv(os.path.join(args.final_csv_path, 'sim_data_pvalues.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Attribution experiment with Baselines')
    parser.add_argument('--dsets', nargs='+', type=str,
                        help='Names of datasets')
    parser.add_argument('--results_dir', type=str,
                        help='directory of where per-experiment result csvs are')
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
