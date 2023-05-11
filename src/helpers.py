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
import seaborn as sns
import pandas as pd
import scipy
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import skfeature
from skfeature.function.similarity_based import lap_score, fisher_score
from skfeature.utility.construct_W import construct_W

from interpretable_tsne.tsne import TSNE, MACHINE_EPSILON, _joint_probabilities_nn


def compute_knn_preservation(dists1, dists2, num_samples, knn=10):
    nbrs1 = NearestNeighbors(n_neighbors=knn, metric='precomputed').fit(squareform(dists1))
    ind1 = nbrs1.kneighbors(return_distance=False)

    nbrs2 = NearestNeighbors(n_neighbors=knn, metric='precomputed').fit(squareform(dists2))
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(num_samples):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    return intersections /num_samples / knn


def get_labels(data_id, data_dir):
    arr_obj_file = '{}/{}/processed_data.npz'.format(data_dir, data_id)
    arr_obj = np.load(arr_obj_file, allow_pickle=True)
    return arr_obj['labels']


def get_data(data_id, data_dir):
    arr_obj_file = '{}/{}/processed_data.npz'.format(data_dir, data_id)
    arr_obj = np.load(arr_obj_file, allow_pickle=True)
    return arr_obj['X_reduced']


def get_fisher_score_features(data_dir, data_id):
    """
    Gets feature-wise Fisher scores.
    """
    # We keep this as-is since the W matrix does not depend on number of neighbours or affinity scores
    # We tested this with the following code:
    #kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': labels, 'k': n_neighbors}
    #W = construct_W(X_reduced, **kwargs)
    #kwargs2 = {"neighbor_mode": "supervised", "fisher_score": True, 'y': labels, 'k': 30}
    #W2 = construct_W(X_reduced, **kwargs2)
    #assert (W.todense() == W2.todense()).all()  # was True
    data = get_data(data_id, data_dir)
    labels = get_labels(data_id, data_dir)
    score = fisher_score.fisher_score(data, labels)
    return score[::-1]


def _get_lap_score_features(data, W):
    scores = lap_score.lap_score(data, W=W, mode='index')
    return scores


def _get_lap_score_features_per_class(data, Ws, labels):
    """
    Gets feature-wise laplace scores, per class.
    """
    scores = []
    for i, label in enumerate(np.unique(labels)):
        class_index = labels == label
        score = _get_lap_score_features(data[class_index], Ws[i])
        scores.append(score)
    return np.stack(scores)


def get_tsne_obj(results_dir, run_id, grad_style):
    results_dir = '{}/{}'.format(results_dir, run_id)
    return np.load(Path(results_dir) / 'tsne_results_style={}.npz'.format(grad_style), allow_pickle=True)


def construct_Q_matrix(embeddings, degrees_of_freedom, machine_epsilon, P, add_diagonal=False):
    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(embeddings, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), machine_epsilon)
    x, y, _ = scipy.sparse.find(P)
    Q = scipy.sparse.csr_matrix((squareform(Q)[x,y], P.indices, P.indptr), shape=(P.shape[0], P.shape[0]))
    if add_diagonal:
        # diagonals are all 0. When we set them to 1 this affects the Laplace Score values a bit
        # default construct_W seems sets the diag values of W to be 1s
        return (Q + scipy.sparse.diags(np.ones(embeddings.shape[0])))
    else:
        return Q


def construct_P_matrix(data, n_jobs, n_neighbors, perplexity, metric, add_diagonal=False):

    # Find the nearest neighbors for every point
    knn = NearestNeighbors(algorithm='auto',
                           n_jobs=n_jobs,
                           n_neighbors=n_neighbors,
                           metric=metric)
    knn.fit(data)
    distances_nn = knn.kneighbors_graph(mode='distance')

    # Free the memory used by the ball_tree
    del knn

    if metric == "euclidean":
        # knn return the euclidean distance but we need it squared
        # to be consistent with the 'exact' method. Note that the
        # the method was derived using the euclidean method as in the
        # input space. Not sure of the implication of using a different
        # metric.
        distances_nn.data **= 2

    # compute the joint probability distribution for the input space
    P, conditional_P, betas = _joint_probabilities_nn(distances_nn,
                                                      perplexity, 0)

    if add_diagonal:
        # diagonals are all 0. When we set them to 1 this affects the Laplace Score values a bit
        # default construct_W seems sets the diag values of W to be 1s
        return (P + scipy.sparse.diags(np.ones(data.shape[0])))
    else:
        return P


def get_affinities_matrix(data, arr_obj, class_index=None, step=250, add_diagonal=False, mode='default'):
    if class_index is None:
        class_index = np.arange(data.shape[0]) # use all the datapoints
    # always run this, we need the values produced by it
    out = arr_obj['out'].item()
    embeddings = out['embeddings'][step]
    metric = 'euclidean'
    degrees_of_freedom = 1
    perplexity = arr_obj['perplexity'].item()
    n_neighbors = min(data[class_index].shape[0] - 1, int(3. * perplexity + 1))
    n_jobs = 4
    if mode == 'default':
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": n_neighbors, 't': 1}
        W = construct_W(data[class_index], **kwargs_W)
    elif mode == 'q_matrix':
        P = construct_P_matrix(data[class_index], n_jobs, n_neighbors, perplexity, metric, add_diagonal) # need the indices
        W = construct_Q_matrix(embeddings[class_index], degrees_of_freedom, MACHINE_EPSILON, P, add_diagonal)
    elif mode == 'p_matrix':
        W = construct_P_matrix(data[class_index], n_jobs, n_neighbors, perplexity, metric, add_diagonal)
    return W


def get_lap_score_features(data, arr_obj, step, add_diagonal, mode='default'):
    W = get_affinities_matrix(data, arr_obj, class_index=None, step=step, add_diagonal=add_diagonal, mode=mode)
    return _get_lap_score_features(data, W)


def get_lap_score_features_per_class(data, labels, arr_obj, step, add_diagonal, mode='default'):
    """
    Gets feature-wise laplace scores, per class.
    """
    Ws = []
    for i, label in enumerate(np.unique(labels)):
        class_index = labels == label
        Ws.append(get_affinities_matrix(data, arr_obj, class_index, step, add_diagonal, mode))
    return _get_lap_score_features_per_class(data, Ws, labels)


def get_explained_variance_ratio(data_dir, data_id):
    arr_obj_file = '{}/{}/processed_data.npz'.format(data_dir, data_id)
    arr_obj = np.load(arr_obj_file, allow_pickle=True)
    X_original = arr_obj['X_original']
    X_reduced = arr_obj['X_reduced']

    pca = PCA(n_components=50, random_state=8)
    X_reduced2 = pca.fit_transform(X_original)
    assert np.allclose(X_reduced2, X_reduced)  # sanity check
    return pca.explained_variance_ratio_


def get_tsne_attributions(seed_value, data_dir, results_dir, run_id, data_id, grad_style, perplexity=30, n_iter=1000, early_exaggeration=4, learning_rate=500):
    # reproducibility (code taken from: https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752)

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    results_dir = '{}/{}'.format(results_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)

    arr_obj_file = '{}/{}/processed_data.npz'.format(data_dir, data_id)
    arr_obj = np.load(arr_obj_file, allow_pickle=True)

    X_reduced = arr_obj['X_reduced']
    #y_train = arr_obj['labels']
    #X_train = arr_obj['X_original']
    #pca_comp = arr_obj['pca_comp']
    #pca_mean = arr_obj['pca_mean']
    #permutation = arr_obj['permutation']

    # check if t-SNE already computed
    if os.path.exists(Path(results_dir) / 'tsne_results_style={}.npz'.format(grad_style)):
        print('File found: {}! Skipping...'.format(Path(results_dir) / 'tsne_results_style={}.npz'.format(grad_style)))
    else:
        # tSNE params
        n_components = 2
        perplexity = perplexity
        verbose = 2
        random_state = seed_value
        n_iter = n_iter
        early_exaggeration = early_exaggeration
        learning_rate = learning_rate
        checkpoint_every = list(np.arange(0, n_iter))
        attr_type = grad_style  # 'kl_obj' 'none'
        init = 'random'
        method = 'barnes_hut'

        ## run tsne, convert to two dimensions
        out = TSNE(n_components=n_components,
                   perplexity=perplexity,
                   random_state=random_state,
                   verbose=verbose,
                   n_iter=n_iter,
                   early_exaggeration=early_exaggeration,
                   learning_rate=learning_rate,
                   checkpoint_every=checkpoint_every,
                   attr=attr_type,
                   init=init,
                   method=method).fit_transform(X_reduced)

        # save results for future analysis
        np.savez(Path(results_dir) / 'tsne_results_style={}.npz'.format(grad_style),
                 out=out,
                 arr_obj_file=arr_obj_file,
                 n_components=n_components,
                 perplexity=perplexity,
                 random_state=random_state,
                 verbose=verbose,
                 n_iter=n_iter,
                 checkpoint_every=checkpoint_every,
                 early_exaggeration=early_exaggeration,
                 learning_rate=learning_rate,
                 attr_type=attr_type,
                 init=init,
                 method=method
                 )


def corrupt_data_using_indices(data, indices, num_feats, method='set_to_0'):
    """
    Corrupts data using indices (sample-wise)
    Method determines how to corrupt the data
    """

    if method == 'set_to_0':
        # Corrupt values in indices
        for i in range(len(data)):
            data[i][indices[i]] = 0
    if method == 'permute':
        data = permute_values(num_feats, indices, data)
    if method == 'mean':
        data = permute_values_using_mean(num_feats, indices, data)
    if method == 'remove':
        assert  (indices[0] == indices[1]).all(), 'Assumes that indices are the same for each sample!'
        data = data[:,np.setdiff1d(np.arange(num_feats), indices[0])]

    return data


def permute_values(num_feats, indices, data):
    """
    Replaces values from indices with feature from a different sample
    So we permute each feature across samples 
    if the sample/feature pair appears in the indices
    """

    to_permute = np.array([data[i][indices[i]] for i in range(len(data))])

    permuted_feats = []
    for feat in range(num_feats):
        permuted_feats.append(np.random.permutation(to_permute[(indices == feat)]))

    # Corrupt values in indices
    for i in range(len(data)):
        data[i][indices[i]] = np.array([permuted_feats[f][0] for f in indices[i]])
        # remove features after!
        for f in indices[i]:
            if len(permuted_feats[f]) > 0:
                permuted_feats[f] = permuted_feats[f][1:]
    return data


def permute_values_using_mean(num_feats, indices, data):
    """
    Replaces values from indices with featurewise mean across indices in sample
    Note that sample mean of each feature is almost 0!
    """
    to_permute = np.array([data[i][indices[i]] for i in range(len(data))])

    permuted_feats = []
    for feat in range(num_feats):
        permuted_feats.append(to_permute[(indices == feat)])
    means = np.array([permuted_feats[i].mean() for i in range(num_feats)])

    # Corrupt values in indices
    for i in range(len(data)):
        data[i][indices[i]] = np.array([means[f] for f in indices[i]])
    return data


def make_cmap_labels(who_labels):
    #  load labels and cmap
    labels_who = np.unique(who_labels)
    labels_who = np.sort(labels_who)
    cmap_who = {labels_who[i]: c for i,c in enumerate(sns.color_palette('husl', labels_who.shape[0]-2)) if ((labels_who[i] != 'Other') or (labels_who[i] != 'Recombinant'))}
    cmap_who['Other'] = 'grey'
    cmap_who['Recombinant'] = 'black'
    return labels_who, cmap_who
