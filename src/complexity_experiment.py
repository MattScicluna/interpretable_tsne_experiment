import os
#import sys
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interpretable_tsne.tsne import TSNE
from set_up_sim_data import gen_sim_data_2_clusters

def main():
    # How algorithm Scales with number of features
    times = []
    times2 = []
    dset_sizes = np.arange(20, 280, 20)
    for dset_size in dset_sizes:
        # generate data
        X_reduced, _ = gen_sim_data_2_clusters(42, effect_size=4, size=1000, num_extra_feats=dset_size-3)

        # fit t-SNE
        start = time.time()

        out = TSNE(n_components=2,
                   perplexity=30,
                   random_state=42,
                   verbose=1,
                   n_iter=251,
                   early_exaggeration=500,
                   learning_rate=500,
                   checkpoint_every=list(np.arange(0, 251)),
                   attr='grad_norm',
                   init='random',
                   method='barnes_hut').fit_transform(X_reduced)

        done = time.time()
        elapsed = done - start
        times.append(elapsed)
        
        start = time.time()
        # t-SNE baseline
        out = TSNE(n_components=2,
                   perplexity=30,
                   random_state=42,
                   verbose=1,
                   n_iter=251,
                   early_exaggeration=500,
                   learning_rate=500,
                   checkpoint_every=list(np.arange(0, 251)),
                   attr='none',
                   init='random',
                   method='barnes_hut').fit_transform(X_reduced)
        done = time.time()
        elapsed2 = done - start
        times2.append(elapsed2)

        print('time with attr: {} without: {}. feats: {:.3f}'.format(elapsed, elapsed2, dset_size))

    print(times)
    print(times2)
    print(dset_sizes)
    np.savez('results/Complexity_exp/feat_complexity_exp', 
             tsne_attr=times,
             tsne=times2,
             dset_sizes=dset_sizes,
             allow_pickle=True)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.margins(0.03)
    ax.grid(True)
    _ = ax.plot(dset_sizes, times2, color='gray', marker='o', linestyle='dashed')
    _ = ax.set_xlabel('Number of Features', fontsize=18)
    _ = ax.set_ylabel('Time (Seconds)', fontsize=18)
    fig.savefig('figures/feat_complexity_graph_tsne', dpi=300)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.margins(0.03)
    ax.grid(True)
    _ = ax.plot(dset_sizes, times, color='gray', marker='o', linestyle='dashed')
    _ = ax.set_xlabel('Number of Features', fontsize=18)
    _ = ax.set_ylabel('Time (Seconds)', fontsize=18)
    fig.savefig('figures/feat_complexity_graph_tsne_attr', dpi=300)
    
    # How algorithm Scales with dataset size
    times = []
    times2 = []
    sample_sizes = np.arange(1000, 20000, 1000)
    for sample_size in sample_sizes:
        # generate data
        X_reduced, _ = gen_sim_data_2_clusters(42, effect_size=4, size=sample_size, num_extra_feats=7)

        # fit t-SNE
        start = time.time()

        out = TSNE(n_components=2,
                   perplexity=30,
                   random_state=42,
                   verbose=1,
                   n_iter=251,
                   early_exaggeration=500,
                   learning_rate=500,
                   checkpoint_every=list(np.arange(0, 251)),
                   attr='grad_norm',
                   init='random',
                   method='barnes_hut').fit_transform(X_reduced)

        done = time.time()
        elapsed = done - start
        times.append(elapsed)
        
        start = time.time()
        # t-SNE baseline
        out = TSNE(n_components=2,
                   perplexity=30,
                   random_state=42,
                   verbose=1,
                   n_iter=251,
                   early_exaggeration=500,
                   learning_rate=500,
                   checkpoint_every=list(np.arange(0, 251)),
                   attr='none',
                   init='random',
                   method='barnes_hut').fit_transform(X_reduced)
        done = time.time()
        elapsed2 = done - start
        times2.append(elapsed2)

        print('time with attr: {} without: {}. sample size: {:.3f}'.format(elapsed, elapsed2, sample_size))

    print(times)
    print(times2)
    print(sample_sizes)
    np.savez('results/Complexity_exp/sample_complexity_exp', 
             tsne_attr=times,
             tsne=times2,
             dset_sizes=sample_sizes,
             allow_pickle=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.margins(0.03)
    ax.grid(True)
    _ = ax.plot(sample_sizes, times2, color='gray', marker='o', linestyle='dashed')
    _ = ax.set_xlabel('Number of Samples', fontsize=18)
    _ = ax.set_ylabel('Time (Seconds)', fontsize=18)
    fig.savefig('figures/sample_complexity_graph_tsne', dpi=300)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.margins(0.03)
    ax.grid(True)
    _ = ax.plot(sample_sizes, times, color='gray', marker='o', linestyle='dashed')
    _ = ax.set_xlabel('Number of Samples', fontsize=18)
    _ = ax.set_ylabel('Time (Seconds)', fontsize=18)
    fig.savefig('figures/sample_complexity_graph_tsne_attr', dpi=300)

    
if __name__ == '__main__':
    main()