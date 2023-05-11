import os
#import sys
from pathlib import Path
#import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


def create_mnist(args):
    data_dir = Path(args.data_dir)

    # Load raw data from https://www.openml.org/d/554
    print('loading MNIST data from {}'.format(data_dir))
    try:
        _path = '{}/{}/processed_data.npz'.format(data_dir, args.id)
        files = np.load(_path, allow_pickle=True)
        X_reduced = files['X_reduced']
        X_original = files['X_original']
        labels = files['labels']
        print('Found processed data at {}. Skipping...'.format(_path))
    except:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, data_home=data_dir)

        # processed data will be put into a new directory
        data_dir = data_dir / str(args.id)
        os.makedirs(data_dir, exist_ok=True)
        random_state = check_random_state(args.seed)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1))

        X /= X.max()  # divide by 255 to get in [0,1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.size, test_size=10, random_state=args.seed)
        print('created subsets')

        #reduce dimensionality to 50 before running tsne
        y_train = pd.Categorical(y_train)

        if args.num_pca != 0:
            pca = PCA(n_components=args.num_pca, random_state=args.seed)
            X_reduced = pca.fit_transform(X_train)
            pca_comp = pca.components_
            pca_mean = pca.mean_
            print('reduced dimension to {} using pca'.format(args.num_pca))
        else:
            X_reduced = X_train
            pca_comp = None
            pca_mean = None
            pca = None

        np.savez(data_dir / 'processed_data',
                 X_reduced=X_reduced,
                 X_original=X_train,
                 labels=y_train,
                 permutation=permutation,
                 seed=args.seed,
                 pca_comp=pca_comp,
                 pca_mean=pca_mean,
                 pca_obj=pca
                 )
        print('saved processed data to {}'.format(data_dir / 'processed_data'))


def main():
    parser = argparse.ArgumentParser(description='Create MNIST (cleaned) dataset')
    parser.add_argument('--data_dir', type=str,
                        help='directory of where to save/load MNIST data from')
    parser.add_argument('--size', type=int, default=10000,
                        help='how large should the dataset be?')
    parser.add_argument('--seed', type=int, default=8,
                        help='random seed (for subsetting, etc...)')
    parser.add_argument('--id', type=int, default=0,
                        help='id of MNIST processing')
    parser.add_argument('--num_pca', type=int, default=50,
                        help='number of pcas to keep for MNIST processing (0 = no pca)')

    args = parser.parse_args()
    create_mnist(args)


if __name__ == '__main__':
    main()
