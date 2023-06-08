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


def gen_sim_data_2_clusters(seed, effect_size=4, size=2000, num_extra_feats=9):
    
    assert size % 2 == 0
    
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    import random
    random.seed(seed)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed)

    data = np.random.randn(size, 1+num_extra_feats)
    label_index = np.random.choice(data.shape[0], size=size//2, replace=False)
    data[label_index, :1] += effect_size  # big difference between both classes!
    classes = np.zeros(shape=(data.shape[0]))
    classes[label_index] = 1
    return data, classes


def gen_sim_data_4_clusters(seed, effect_size_main=4, effect_size_subgroups=3, size=2000, num_extra_feats=7):
    
    assert size % 4 == 0
    
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    import random
    random.seed(seed)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed)

    data = np.random.randn(size, 3+num_extra_feats)
    #data = np.zeros(shape=(2000, 10))
    label_index = np.random.choice(data.shape[0], size=size//2, replace=False)
    label_index_opposite = np.setdiff1d(np.arange(size), label_index)

    data[label_index, :1] += effect_size_main # big difference between both classes!

    data[label_index[label_index.shape[0]//2:], 1] += effect_size_subgroups # each class has a subclass
    data[label_index_opposite[label_index_opposite.shape[0]//2:], 2] += effect_size_subgroups # each class has a subclass

    classes = np.zeros(shape=(data.shape[0]))
    classes[label_index[:label_index.shape[0]//2]] = 1
    classes[label_index_opposite[label_index_opposite.shape[0]//2:]] = 2
    classes[label_index_opposite[:label_index_opposite.shape[0]//2]] = 3
    return data, classes


def create_sim_data(args):

    for seed in range(10):
        print('Creating 2 Cluster Sim Data')
        for effect_size in [1, 2, 3, 4, 6, 8]:
            X, y = gen_sim_data_2_clusters(seed, effect_size)
            data_dir = Path(args.data_dir) / 'Sim_2_Clusters_effect={}'.format(effect_size) / str(seed)
            if os.path.exists(data_dir):
                print('Found processed data at {}. Skipping...'.format(data_dir / 'processed_data'))
            else:
                os.makedirs(data_dir, exist_ok=True)
                np.savez(data_dir / 'processed_data',
                         X_reduced=X,
                         X_original=X,
                         labels=y,
                         seed=seed
                         )
                print('saved processed data to {}'.format(data_dir / 'processed_data'))

        print('Creating 4 Cluster Sim Data')
        for effect_size in [1, 2, 4, 6, 8]:
            for effect_size2 in [1, 2, 3, 5, 7]:
                X, y = gen_sim_data_4_clusters(seed, effect_size, effect_size2)
                data_dir = Path(args.data_dir)  / 'Sim_4_Clusters_effect1={}_effect2={}'.format(effect_size, effect_size2) / str(seed)
                if os.path.exists(data_dir):
                    print('Found processed data at {}. Skipping...'.format(data_dir / 'processed_data'))
                else:
                    os.makedirs(data_dir, exist_ok=True)
                    np.savez(data_dir / 'processed_data',
                             X_reduced=X,
                             X_original=X,
                             labels=y,
                             seed=seed
                             )
                    print('saved processed data to {}'.format(data_dir / 'processed_data'))


def main():
    parser = argparse.ArgumentParser(description='Create Simulated datasets')
    parser.add_argument('--data_dir', type=str,
                        help='directory of where to save/load Simulated data from')
    args = parser.parse_args()
    create_sim_data(args)


if __name__ == '__main__':
    main()
