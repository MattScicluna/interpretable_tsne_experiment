"""
Helper functions for attribution visualizations
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim import models



### 20 NG helper functions

def remove_min_max_attrs(attr, min_percentile_remove, max_percentile_remove):

    if min_percentile_remove == 0 and max_percentile_remove == 100:
        return attr  # Nothing needs to be done!
    else:
        perc_u = np.percentile(np.abs(attr), min_percentile_remove)
        perc_l = np.percentile(np.abs(attr), max_percentile_remove)
        mask = ~((perc_l < np.abs(attr)) | (np.abs(attr) > perc_u))

        if mask.sum() == attr.shape[1]:
            # removed all the pixels! Need to override
            # in this case we just return the pixels that are the max attr value!
            mask = attr != attr.max()
        return np.ma.masked_array(data=attr, mask=mask)

def transform_attr(attr_plot, model, words):
    """
    Applies transformation on attributions

    Parameters
    ----------
    attr_plot: np.array
        array of size (n_sampes, pca_dims)

    Returns
    =======
    attr_plot: np.array
        array of size (n_sampes, input_dims**)
    """

    # recover dims
    word_embeddings = [model[doc] for doc in words]

    attr_plot = np.array([(np.stack(word_embeddings)@(attr_plot).T)])

    return np.abs(attr_plot)

def process_attr(attrs, attr_mapping, min_percentile_remove, max_percentile_remove):
    """
    cleans attr prior to plotting
    """

    attr = np.take(np.concatenate([attrs, np.zeros(shape=(1,1))], 1),  # zero attribution for UNK values
                    attr_mapping.filled(), axis=None)

    attr = remove_min_max_attrs(attr, min_percentile_remove, max_percentile_remove)

    return attr

def _textplot(sent, attr_mapping, attr_vals, ax, aspect, **kwargs):
    img = np.ma.masked_array(data=np.zeros(shape=attr_mapping.shape),
                                mask=True)
    _ = ax.imshow(img, cmap='Reds', interpolation='none', aspect=aspect)

    #  |attrs| > white_thresh will have white text to be visible!
    white_thresh = np.percentile(np.abs(attr_vals), 90)

    for j, line in enumerate(list(filter(None, sent.split('\n')))):
        for i, w in enumerate(line):
            if np.ma.is_masked(attr_mapping[j, i]):
                # word did not appear in vocab. Make words greyed out
                ax.text(i, j, w, ha="center", va="center", fontdict={'family': 'serif', 'color':  'grey'}, **kwargs)
            elif np.abs(np.abs(attr_vals)[j, i]) > white_thresh:
                ax.text(i, j, w, ha="center", va="center", fontdict={'family': 'serif', 'color':  'white'}, **kwargs)
            else:
                ax.text(i, j, w, ha="center", va="center", fontdict={'family': 'serif', 'color':  'black'}, **kwargs)

    ax.axis('off')

def plot_attr_20ng(data, words, attrs, idx, model, attr_mapping, ax, min_percentile_remove=0, max_percentile_remove=100, aspect=5):
    """
    Plot attributions for 20NG
    """

    raw_sent = data[idx]
    sent = words[idx]
    attr_plot = attrs[idx]
    mapping = attr_mapping[idx]

    attr_plot = transform_attr(attr_plot, model, sent)

    #  fill attr_mapping using values from attr_to_plot
    attr_img = process_attr(attr_plot,
                            mapping,
                            min_percentile_remove=min_percentile_remove,
                            max_percentile_remove=max_percentile_remove,
                            )

    #  plot text first
    _textplot(raw_sent, mapping, attr_img, ax, aspect)

    ax.imshow(attr_img, alpha=1, cmap='Reds', vmin=attr_img.min(), vmax=attr_img.max(), interpolation='nearest', aspect=aspect)
    ax.axis('off')
