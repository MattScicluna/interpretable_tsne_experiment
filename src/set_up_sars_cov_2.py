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


def _remove_columns(data, positions, remove_first, remove_last):
    bool_index = ((positions > remove_first) & (positions < 29903 - remove_last)) # 29903 = size of genome
    data = data.loc[:, bool_index]
    return data


def _get_indices_that_startswith(metadata, startswith, lineage_name, col_to_use, col_to_change):
    contains_indices = metadata[col_to_use].str.startswith(startswith)
    metadata[col_to_change].loc[contains_indices] = lineage_name
    return metadata


def _convert_string_date_to_numeric(string_date):

    times = pd.to_datetime(string_date, errors='coerce')
    times_float = np.array((times - times.min()).dt.days/(times.max() - times.min()).days)
    return times_float


def _make_who_labels(metadata):
    metadata['WHO Label'] = np.nan
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.7', 'Alpha', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.28.1', 'Gamma', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.617.2', 'Delta', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.351', 'Beta', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.529.1', 'Omicron BA.1', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.529.2', 'Omicron BA.2', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.529.4', 'Omicron BA.4', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.529.5', 'Omicron BA.5', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.529.5.3.1.1.1.1', 'Omicron BQ', 'Full PANGO Transform', 'WHO Label')
    metadata = _get_indices_that_startswith(metadata, 'X', 'Recombinant', 'PANGO ID', 'WHO Label')
    metadata['WHO Label'].loc[metadata['WHO Label'].isna()] = 'Other'

    metadata['WHO Label Detailed'] = metadata['WHO Label']
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.529.5.1', 'Omicron BA.5.1', 'Full PANGO Transform', 'WHO Label Detailed')
    metadata = _get_indices_that_startswith(metadata, 'B.1.1.529.5.2', 'Omicron BA.5.2', 'Full PANGO Transform', 'WHO Label Detailed')

    return metadata


def create_sars_cov_2(args):

    data_dir = Path(args.data_dir) / str(args.id)
    os.makedirs(data_dir, exist_ok=True)
    print('loading SARS-CoV-2 data from {}'.format(data_dir))
    try:
        _path = data_dir / 'processed_data.npz'
        files = np.load(_path, allow_pickle=True)
        X_reduced = files['X_reduced']
        X_original = files['X_original']
        metadata = files['metadata']
        print('Found processed data at {}. Skipping...'.format(_path))
    except:
        # processed data will be put into a new directory
        random_state = check_random_state(args.seed)
        print('Creating SARS-CoV-2 data and saving it to: {}'.format(data_dir))

        # Sparsify data
        sparse_data_name = os.path.join(data_dir, 'sparse_{}'.format(args.raw_data_name))
        raw_data_name = os.path.join(args.raw_data_path, args.raw_data_name + '.mat')
        raw_data_colnames = os.path.join(args.raw_data_path, args.raw_data_name + '.list')

        # load sequence data from .mat file
        if not os.path.exists(sparse_data_name):
            print('Could not find sparse data file: {}'.format(sparse_data_name))
            print('Sparsifying data from {}...'.format(raw_data_name))

            # Read data
            # Specify types for faster parsing and to save memory
            #col_names = ['POS'] + [str(i) for i in range(1, 29904)]
            col_names = pd.read_csv(raw_data_colnames, sep='\t').columns.to_numpy()
            #types = [str] + [np.int8] * (len(col_names) - 1)
            types = [np.int8] * (len(col_names))

            #data = pd.read_csv(mat_fname, dtype=dict(zip(col_names, types)), sep='\t', header=0, index_col='POS', chunksize=args.chunk_size)
            X = pd.read_csv(raw_data_name, dtype=dict(zip(col_names, types)), sep='\t', header=0, index_col=0, chunksize=1000)

            n = 0
            result = []
            for chunk in X:
                result.append(chunk.astype(pd.SparseDtype(np.int8, fill_value=0)))
                n += chunk.shape[0]
                print(f'Processed {n} rows')

            sdf = pd.concat(result)
            sdf.index.name = 'POS' # set index name
            sdf.to_pickle(sparse_data_name)

            print(f'Sparse dataframe has density {sdf.sparse.density}...')
            print(f'Sparse dataframe has shape   {sdf.shape}...')

        # Load from file!
        with open(sparse_data_name, "rb") as fh:
            X = pickle.load(fh)
            print('loaded sparse raw data from: {}'.format(sparse_data_name))

        # Replace N with S in index (for some reason it was changed)
        X.index = X.index.map(lambda x: x.replace('N', 'S'))

        # Add position column
        positions = np.array(np.array(X.columns.str.split('_').to_list())[:,0], dtype=int)

        # load metadata file
        with open(args.metadata_path) as fh:
            metadata = pd.read_csv(fh, sep='\t', header=None)

        metadata = metadata.drop(columns=[0, 4, 5, 6, 7, 8, 13, 15, 24, 25])
        metadata = metadata.set_index(1)
        metadata.columns = ['UNK Date', 
                            'UNK Continent', 
                            'UNK Date 2', 
                            'UNK Continent 2', 
                            'Country', 
                            'Region', 
                            'UNK Numeric', 
                            'Age', 
                            'Gender', 
                            'GISAID Clade', 
                            'PANGO ID', 
                            'PANGO Label Type', 
                            'Long Name', 
                            'Long Name 2', 
                            'UNK Date 3', 
                            'UNK Boolean', 
                            'UNK Boolean 2', 
                            'UNK Numeric 2', 
                            'UNK Numeric 3', 
                            'Haplotype 22', 
                            'Haplotype 25', 
                            'Full PANGO Transform']

        # remove NAs
        metadata.loc[metadata['PANGO ID'].isna(), 'PANGO ID'] = 'None'
        metadata.loc[metadata['Full PANGO Transform'].isna(), 'Full PANGO Transform'] = 'None'
        metadata.loc[metadata['Haplotype 22'].isna(), 'Haplotype 22'] = 'Other'
        metadata.loc[metadata['Haplotype 25'].isna(), 'Haplotype 25'] = 'Other'

        old_shape = X.shape
        X = _remove_columns(X, positions, args.remove_first, args.remove_last)
        print('Data went from shape {} to {}'.format(old_shape, X.shape))
        X = X.astype(np.float32)  # Convert to float32 while sparse (otherwise numpy/sklearn will likely do it for us on the dense array)

        metadata = _make_who_labels(metadata)

        # cast time as numeric between 0 and 1
        times = metadata['UNK Date']

        #  If date unresolved, assume 1st of the month
        times = times.str.replace('XX', '01')
        times = times.str.replace('00', '01')

        times_float = _convert_string_date_to_numeric(times)
        metadata['Date_float']  = times_float

        metadata['num_months'] = pd.to_datetime(times).dt.month + (pd.to_datetime(times).dt.year -2020)*12
        print('Added time-based metadata columns')

        #reduce dimensionality to 50 before running tsne
        if args.num_pca != 0:
            pca = PCA(n_components=args.num_pca, random_state=args.seed)
            X_reduced = pca.fit_transform(X.to_numpy())
            pca_comp = pca.components_
            pca_mean = pca.mean_
            print('reduced dimension to {} using pca'.format(args.num_pca))
        else:
            X_reduced = X_train
            pca_comp = None
            pca_mean = None

        np.savez(data_dir / 'processed_data',
                 X_reduced=np.array(X_reduced, dtype=np.double), # need double for t-SNE attribution code!
                 X_original=X,
                 X_index=X.index.values,
                 X_colnames=X.columns.values,
                 metadata=metadata,
                 metadata_colnames=metadata.columns.values,
                 seed=args.seed,
                 pca_comp=pca_comp,
                 pca_mean=pca_mean,
                 pca_obj=pca
                 )
        print('saved processed data to {}'.format(data_dir / 'processed_data'))


def main():
    parser = argparse.ArgumentParser(description='Create SARS-CoV-2 sequence dataset')
    parser.add_argument('--data_dir', type=str,
                        help='directory of where to save/load SARS-CoV-2 sequence data from')
    parser.add_argument('--raw_data_path', type=str,
                        help='path where to load SARS-CoV-2 sequence .mat file from')
    parser.add_argument('--raw_data_name', type=str,
                        help='name of SARS-CoV-2 sequence .mat file from')
    parser.add_argument('--metadata_path', type=str,
                        help='full path where to load SARS-CoV-2 metadata .csv file from')
    parser.add_argument('--seed', type=int, default=8,
                        help='random seed (for subsetting, etc...)')
    parser.add_argument('--remove_first', type=int, default=100,
                        help='remove first X positions')
    parser.add_argument('--remove_last', type=int, default=100,
                        help='remove last X positions')
    parser.add_argument('--id', type=int, default=0,
                        help='id of SARS-CoV-2 processing')
    parser.add_argument('--num_pca', type=int, default=100,
                        help='number of pcas to keep for SARS-CoV-2 processing (0 = no pca)')

    args = parser.parse_args()
    create_sars_cov_2(args)


if __name__ == '__main__':
    main()
