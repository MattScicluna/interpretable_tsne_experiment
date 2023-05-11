from helpers import *


def load_tsne(seed_value, data_dir, tsne_output_dir, run_id, data_id, data_id_new, indices, method):

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

    try:
        arr_obj_file = '{}/{}/{}_processed_data.npz'.format(data_dir, data_id, data_id_new)
        arr_obj = np.load(arr_obj_file, allow_pickle=True)
        X_reduced = arr_obj['X_reduced']
        #y_train = pd.Categorical(arr_obj['labels'])
        #X_train = arr_obj['X_original']
        #pca_comp = arr_obj['pca_comp']
        #pca_mean = arr_obj['pca_mean']
        #permutation = arr_obj['permutation']
        print('loaded corrupted processed data from {}/{}/{}_processed_data.npz'.format(data_dir, data_id, data_id_new))
    except FileNotFoundError:
        # Load uncorrupted data
        arr_obj_file = '{}/{}/processed_data.npz'.format(data_dir, data_id)
        arr_obj = np.load(arr_obj_file, allow_pickle=True)
        X_reduced = arr_obj['X_reduced']
        #y_train = pd.Categorical(arr_obj['labels'])
        #X_train = arr_obj['X_original']
        #pca_comp = arr_obj['pca_comp']
        #pca_mean = arr_obj['pca_mean']
        #permutation = arr_obj['permutation']

        X_reduced = corrupt_data_using_indices(X_reduced, indices, X_reduced.shape[1], method=method)

        #os.makedirs('{}/{}'.format(data_dir, data_id_new), exist_ok=True)
        np.savez('{}/{}/{}_processed_data.npz'.format(data_dir, data_id, data_id_new),
                 X_reduced=X_reduced,
                 seed=seed_value)
        # Save Space!
        #         X_original=X_train,
        #         labels=y_train,
        #         permutation=permutation,
        #         pca_comp=pca_comp,
        #         pca_mean=pca_mean
        #         )
        print('saved corrupted processed data to {}/{}/{}_processed_data.npz'.format(data_dir, data_id, data_id_new))

        # Now this is the array object file!
        arr_obj_file = '{}/{}/{}_processed_data.npz'.format(data_dir, data_id, data_id_new)

    # hack
    #if len(pca_comp.shape) == 0:
    #    pca_comp = None

    # tSNE params
    n_components = 2
    perplexity = 30
    verbose = 2
    random_state = seed_value
    n_iter = 1000
    early_exaggeration = 4
    learning_rate = 500
    checkpoint_every = list(np.arange(0, 1000))  # list(np.arange(0, 1000, 20)) #[]
    attr_type = 'none'  # 'none'
    init = 'random'
    method = 'barnes_hut'

    try:
        arr_obj = np.load(Path(tsne_output_dir) / 'tsne_results_{}.npz'.format(data_id_new), allow_pickle=True)
        out = arr_obj['out'].item()
        print('Loaded t-SNE fit on corrupted data from here: {}'.format(Path(tsne_output_dir) / 'tsne_results_{}.npz'.format(data_id_new)))
    except FileNotFoundError:
        print('Did not find {}. Running t-SNE now'.format(Path(tsne_output_dir) / 'tsne_results_{}.npz'.format(data_id_new)))

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
        np.savez(Path(tsne_output_dir) / 'tsne_results_{}'.format(data_id_new),
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
    return out, X_reduced


def make_metrics(out,
                 dists_ref,
                 dists_real,
                 seed_value,
                 label_tsne_orig,
                 label_true, 
                 r_indices,
                 knn=10):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    Y = out['embeddings'][-1]

    # Spearman Correlation btw t-SNE's
    dists = scipy.spatial.distance.pdist(Y, metric='euclidean')
    corr = scipy.stats.spearmanr(dists[r_indices], dists_ref[r_indices]).correlation

    # KNN preservation score
    knn_preserve = compute_knn_preservation(dists_ref, dists, num_samples=Y.shape[0], knn=knn)

    # Adjusted RAND index
    # make labels
    #labels_pred = DBSCAN(eps=0.5, min_samples=500, metric='euclidean').fit_predict(dists)
    kmeans = KMeans(n_clusters=10, init='k-means++').fit(Y)
    labels_pred = kmeans.labels_
    rand_index = adjusted_rand_score(label_tsne_orig, labels_pred)

    # TRUE TO THE DATA???
    # lets see what happens if we do these measurements but using the actual data info
    corr_ground_truth = scipy.stats.spearmanr(dists_real[r_indices], dists[r_indices]).correlation
    rand_index_ground_truth = adjusted_rand_score(label_true, labels_pred)
    knn_preserve_ground_truth = compute_knn_preservation(dists_real, dists, num_samples=Y.shape[0], knn=knn)

    return corr, knn_preserve, rand_index, corr_ground_truth, rand_index_ground_truth, knn_preserve_ground_truth


def make_top_attr_indices(num_feats_keep, step, attrs):
    """
    select indices based on top attributions
    """
    attrs = attrs[step]  # look at attributions after early exaggeration only!
    top_indices = np.argsort(np.abs(attrs), axis=1)[:,num_feats_keep:]
    return top_indices


def make_random_indices(num_feats, num_feats_corrupted, num_samples, seed):
    """
    Select indices randomly. Each sample gets its own random indices
    """
    np.random.seed(seed)  # make sure that the random indices are different between runs!
    return np.stack([np.random.choice(np.arange(num_feats), replace=False, size=num_feats_corrupted) for i in range(num_samples)])


def make_indices_from_score(score, num_feats_corrupted, num_samples):
    """
    repeats indices based on scoring (duplicates across samples)
    """
    return np.stack([score[:num_feats_corrupted] for i in range(num_samples)])


def make_class_indices_from_score(score_per_class, num_feats_corrupted, num_samples, labels):
    """
    repeats indices based on scoring and classes (duplicates across samples in the same class)
    Each class (labels) has a different score (row in score_per_class) that is repeated
    """
    final_indices = np.zeros(shape=(num_samples, num_feats_corrupted), dtype=np.long)

    for i, label in enumerate(np.unique(labels)):
        class_index = labels == label
        _tmp = make_indices_from_score(score_per_class[i], num_feats_corrupted, class_index.sum())
        final_indices[class_index] = _tmp
    return final_indices


def make_unif_indices_from_attr(indices, num_feats_keep, num_feats, num_samples):
    """
    Finds features with highest attributions and returns these indices, duplicated across samples
    """
    counts = []
    for i in range(num_feats):
        counts.append(np.count_nonzero(indices == i))
    return np.array([np.argsort(np.array(counts))[num_feats_keep:] for i in range(num_samples)])


def make_unif_class_indices_from_attr(indices, num_feats_keep, num_feats, num_samples, labels):
    final_indices = np.zeros(shape=(num_samples, num_feats-num_feats_keep), dtype=np.long)
    indices_per_class = np.zeros(shape=(len(np.unique(labels)), num_feats-num_feats_keep), dtype=np.long)

    for j, label in enumerate(np.unique(labels)):
        counts = []
        class_index = labels == label
        for i in range(num_feats):
            counts.append(np.count_nonzero(indices[class_index] == i))
        class_idxs = np.argsort(np.array(counts))[num_feats_keep:]
        final_indices[class_index] = np.array([class_idxs for i in range(class_index.sum())])
        indices_per_class[j] = class_idxs
    return final_indices, indices_per_class


def make_top_attr_times_input_indices(num_feats_keep, step, attrs, data_dir, data_id):
    """
    Same as make_top_attr_indices but uses input times attribution
    """
    data = get_data(data_id, data_dir)
    return np.argsort(np.abs((attrs[step]*data)), axis=1)[:,num_feats_keep:]


def make_top_attr_times_input_ge0_indices(num_feats_keep, step, attrs, data_dir, data_id):
    """
    Same as make_top_attr_indices but uses input times attribution
    """
    data = get_data(data_id, data_dir)
    _attr = attrs[step]*(attrs[step] > 0)
    _data = data*(data > 0)
    return np.argsort(_attr*_data, axis=1)[:,num_feats_keep:]


def make_indices_from_feat_size(num_feats_keep, data_dir, data_id):
    """
    Computes attribution based on size of input feature. Used as another control
    """
    data = get_data(data_id, data_dir)

    # select indices based on feature size
    top_indices = np.argsort(np.abs(data), axis=1)[:,num_feats_keep:]
    return top_indices


def make_top_attr_ge0_indices(num_feats_keep, step, attrs):
    """
    select indices based on top attributions (only >0)
    """
    attrs = attrs[step]*(attrs[step] > 0)
    top_indices = np.argsort(attrs, axis=1)[:,num_feats_keep:]
    return top_indices


def get_indices(indices_name,
                attrs,
                num_feats_keep,
                args,
                level,
                num_feats_corrupted=None,
                lscore=None,
                lscore_p_matrix=None,
                lscore_q_matrix=None,
                fsscore=None,
                labels=None,
                r_run=None):

    removed_indices = 'NA'
    num_feats = num_feats_keep+num_feats_corrupted  # need to add for legacy reasons
    if level == 'individual':
        if indices_name == 'top_attr':
            top_indices = make_top_attr_indices(num_feats_keep, step=args.step, attrs=attrs)
        elif indices_name == 'attr_feat':
            top_indices = make_top_attr_times_input_indices(num_feats_keep, args.step, attrs, args.data_dir, args.data_id)
        elif indices_name == 'attr_ge_0_feat_ge_0':
            top_indices = make_top_attr_times_input_ge0_indices(num_feats_keep, args.step, attrs, args.data_dir, args.data_id)
        elif indices_name == 'top_attr_ge_0':
            top_indices = make_top_attr_ge0_indices(num_feats_keep, step=args.step, attrs=attrs)
        elif indices_name == 'feat_size':
            top_indices = make_indices_from_feat_size(num_feats_keep, args.data_dir, args.data_id)
        elif indices_name == 'random':
            top_indices = make_random_indices(num_feats, num_feats_corrupted, attrs.shape[1], r_run)
        else:
            raise ValueError
    elif level == 'class':
        if indices_name == 'random':
            random_indices_class = make_random_indices(num_feats, num_feats_corrupted, attrs.shape[1], r_run)[:10]
            top_indices = make_class_indices_from_score(random_indices_class, num_feats_corrupted, attrs.shape[1], labels)
            removed_indices = json.dumps(random_indices_class.tolist())
        elif indices_name == 'top_ls_class':
            top_indices =  make_class_indices_from_score(lscore, num_feats_corrupted, attrs.shape[1], labels)
            removed_indices = json.dumps(lscore[:, :num_feats_corrupted].tolist())
        elif indices_name == 'top_ls_class_q_matrix':
            top_indices =  make_class_indices_from_score(lscore_q_matrix, num_feats_corrupted, attrs.shape[1], labels)
            removed_indices = json.dumps(lscore_p_matrix[:, :num_feats_corrupted].tolist())
        elif indices_name == 'top_ls_class_p_matrix':
            top_indices =  make_class_indices_from_score(lscore_p_matrix, num_feats_corrupted, attrs.shape[1], labels)
            removed_indices = json.dumps(lscore_p_matrix[:, :num_feats_corrupted].tolist())
        elif indices_name == 'top_attr_times_feat_unif_class':
            attr_times_feat_indices = make_top_attr_times_input_indices(num_feats_keep, args.step, attrs, args.data_dir, args.data_id)
            top_indices, removed_indices = make_unif_class_indices_from_attr(attr_times_feat_indices, num_feats_keep, num_feats, attrs.shape[1], labels)
            removed_indices = json.dumps(removed_indices.tolist())
        elif indices_name == 'top_attr_unif_class':
            attr_indices = make_top_attr_indices(num_feats_keep, step=args.step, attrs=attrs)
            top_indices, removed_indices = make_unif_class_indices_from_attr(attr_indices, num_feats_keep, num_feats, attrs.shape[1], labels)
            removed_indices = json.dumps(removed_indices.tolist())
        elif indices_name == 'top_feat_class':
            feat_indices = make_indices_from_feat_size(num_feats_keep, args.data_dir, args.data_id)
            top_indices, removed_indices = make_unif_class_indices_from_attr(feat_indices, num_feats_keep, num_feats, attrs.shape[1], labels)
            removed_indices = json.dumps(removed_indices.tolist())
        else:
            raise ValueError
    elif level == 'global':
        if indices_name == 'random':
            random_indices_global = make_random_indices(num_feats, num_feats_corrupted, attrs.shape[1], r_run)[0]
            top_indices = make_indices_from_score(random_indices_global, num_feats_corrupted, attrs.shape[1])
        elif indices_name == 'top_pc':
            top_indices = make_indices_from_score(np.arange(num_feats), num_feats_corrupted, attrs.shape[1])
        elif indices_name == 'top_ls':
            top_indices = make_indices_from_score(lscore, num_feats_corrupted, attrs.shape[1])
        elif indices_name == 'top_ls_q_matrix':
            top_indices = make_indices_from_score(lscore_q_matrix, num_feats_corrupted, attrs.shape[1])
        elif indices_name == 'top_ls_p_matrix':
            top_indices = make_indices_from_score(lscore_p_matrix, num_feats_corrupted, attrs.shape[1])        
        elif indices_name == 'top_fs':
            top_indices = make_indices_from_score(fsscore, num_feats_corrupted, attrs.shape[1])
        elif indices_name == 'top_attr_times_feat_unif':
            attr_times_feat_indices = make_top_attr_times_input_indices(num_feats_keep, args.step, attrs, args.data_dir, args.data_id)
            top_indices = make_unif_indices_from_attr(attr_times_feat_indices, num_feats_keep, num_feats, attrs.shape[1])
        elif indices_name == 'top_attr_unif':
            attr_indices = make_top_attr_indices(num_feats_keep, step=args.step, attrs=attrs)
            top_indices = make_unif_indices_from_attr(attr_indices, num_feats_keep, num_feats, attrs.shape[1])
        elif indices_name == 'top_feat':
            feat_indices = make_indices_from_feat_size(num_feats_keep, args.data_dir, args.data_id)
            top_indices = make_unif_indices_from_attr(feat_indices, num_feats_keep, num_feats, attrs.shape[1])
        else:
            raise ValueError
        removed_indices = json.dumps(top_indices[0, :num_feats_corrupted].tolist()) # they are all the same, so just use the first!
    else:
        raise ValueError
    return top_indices, removed_indices


def remove_with_index(args,
                      df,
                      tsne_output_dir,
                      new_id,
                      label_tsne_orig,
                      label_true,
                      dists_ref,
                      dists_real,
                      top_indices,
                      r_indices,
                      indices,
                      num_feats_corrupted,
                      num_feats,
                      grad_style,
                      removed_indices,
                      step):

    out, X_reduced = load_tsne(args.run_id,
                               args.data_dir,
                               tsne_output_dir,
                               args.run_id,
                               args.data_id,
                               new_id,
                               top_indices,
                               args.method)

    corr, knn_preserve, rand_index, corr_ground_truth, rand_index_ground_truth, knn_preserve_ground_truth = make_metrics(out,
                 dists_ref,
                 dists_real,
                 args.run_id,
                 label_tsne_orig,
                 label_true, 
                 r_indices,
                 knn=10)

    next_row = {'Correlation': corr,
                'knn_preservation': knn_preserve,
                'ari': rand_index,
                'Correlation gt': corr_ground_truth,
                'knn_preservation gt': knn_preserve_ground_truth,
                'ari gt': rand_index_ground_truth,
                'Index': indices,
                '% Corrupted': num_feats_corrupted/num_feats,
                'Run_id': args.run_id,
                'grad_style': grad_style,
                'method': args.method,
                'step': step,
                'removed_indices': removed_indices}

    df = pd.concat([df, pd.DataFrame(next_row, index=[0])])

    return df

def remove_each_step(args,
                     df,
                     num_feats,
                     num_feats_corrupted,
                     attrs,
                     tsne_output_dir,
                     label_tsne_orig,
                     label_true,
                     lscore, 
                     lscore_p_matrix,
                     lscore_q_matrix,
                     fsscore,
                     dists_ref,
                     dists_real,
                     r_indices,
                     level='individual'):

    num_feats_keep = num_feats - num_feats_corrupted

    for indices in args.indices_list:
        top_indices, removed_indices = get_indices(indices, 
                                                   attrs, 
                                                   num_feats_keep, 
                                                   args, 
                                                   level, 
                                                   num_feats_corrupted, 
                                                   lscore, 
                                                   lscore_p_matrix,
                                                   lscore_q_matrix,
                                                   fsscore,
                                                   label_true, 
                                                   None)


        if level == 'individual':
            if indices in ['feat_size']:
                # attribution style does not affect the results!
                new_id = str(args.run_id)+'_{}_rem={}_step={}_method={}'.format(indices, num_feats_corrupted, args.step, args.method)
                grad_style = 'NA'
                step = 'NA'
            else:
                new_id = str(args.run_id)+'_{}_rem={}_step={}_method={}_style={}'.format(indices, num_feats_corrupted, args.step, args.method, args.grad_style)
                grad_style = args.grad_style
                step = args.step
        elif level == 'class':
            if indices in ['top_ls_class', 'top_ls_class_p_matrix', 'top_ls_class_q_matrix', 'top_feat_class']:
                # attribution style does not affect the results!
                new_id = str(args.run_id)+'_{}_rem={}_step={}_method={}'.format(indices, num_feats_corrupted, args.step, args.method)
                grad_style = 'NA'
                step = 'NA'
            else:
                new_id = str(args.run_id)+'_{}_rem={}_step={}_method={}_style={}'.format(indices, num_feats_corrupted, args.step, args.method, args.grad_style)
                grad_style = args.grad_style
                step = args.step
        elif level == 'global':
            if indices in ['top_pc', 'top_fs', 'top_ls_q_matrix', 'top_ls_p_matrix', 'top_ls', 'top_feat']:
                # attribution style does not affect the results!
                new_id = str(args.run_id)+'_{}_rem={}_step={}_method={}'.format(indices, num_feats_corrupted, args.step, args.method)
                grad_style = 'NA'
                step = 'NA'
            else:
                new_id = str(args.run_id)+'_{}_rem={}_step={}_method={}_style={}'.format(indices, num_feats_corrupted, args.step, args.method, args.grad_style)
                grad_style = args.grad_style
                step = args.step
        else:
            raise ValueError

        df = remove_with_index(args,
                               df,
                               tsne_output_dir,
                               new_id,
                               label_tsne_orig,
                               label_true,
                               dists_ref,
                               dists_real,
                               top_indices,
                               r_indices,
                               indices,
                               num_feats_corrupted,
                               num_feats,
                               grad_style,
                               removed_indices,
                               step)

    # do 10 random runs
    for r_run in range(10):
        rand_indices, removed_indices =get_indices('random', 
                                                   attrs, 
                                                   num_feats_keep, 
                                                   args, 
                                                   level, 
                                                   num_feats_corrupted, 
                                                   lscore, 
                                                   lscore_p_matrix,
                                                   lscore_q_matrix,
                                                   fsscore,
                                                   label_true, 
                                                   r_run)
                
        if level == 'individual':
            new_id = str(args.run_id)+'_rand_rem={}_run={}_step={}_method={}'.format(num_feats_corrupted, r_run, args.step, args.method)
        elif level == 'class':
            new_id = str(args.run_id)+'_rand_class_rem={}_run={}_step={}_method={}'.format(num_feats_corrupted, r_run, args.step, args.method)
        elif level == 'global':
            new_id = str(args.run_id)+'_rand_group_rem={}_run={}_step={}_method={}'.format(num_feats_corrupted, r_run, args.step, args.method)

        df = remove_with_index(args,
                               df,
                               tsne_output_dir,
                               new_id,
                               label_tsne_orig,
                               label_true,
                               dists_ref,
                               dists_real,
                               rand_indices,
                               r_indices,
                               'random',
                               num_feats_corrupted,
                               num_feats,
                               grad_style,
                               removed_indices,
                               step)

    # save afer each iteration!
    print(df)
    if level == 'individual':
        df.to_csv(os.path.join(args.final_csv_path, 'attr_exp_results_individual_feats_{}.csv'.format(args.run_id)))
    elif level == 'class':
        df.to_csv(os.path.join(args.final_csv_path, 'attr_exp_results_class_feats_{}.csv'.format(args.run_id)))
    elif level == 'global':
        df.to_csv(os.path.join(args.final_csv_path, 'attr_exp_results_global_feats_{}.csv'.format(args.run_id)))

    return df


def remove_all_steps(args, level='individual'):

    # check that indices are valid:
    VALID_LIST = {}
    VALID_LIST['individual'] = ['top_attr', 'top_attr_ge_0', 'attr_feat', 'attr_ge_0_feat_ge_0', 'feat_size']
    VALID_LIST['class'] = ['top_ls_class', 'top_ls_class_p_matrix', 'top_ls_class_q_matrix', 'top_attr_times_feat_unif_class', 'top_attr_unif_class', 'top_feat_class']
    VALID_LIST['global'] = ['top_pc', 'top_fs', 'top_ls', 'top_ls_p_matrix', 'top_ls_q_matrix', 'top_attr_times_feat_unif', 'top_attr_unif', 'top_feat']

    for index in args.indices_list:
        assert index in VALID_LIST[level]
    
    # Create dataframe
    df = pd.DataFrame({'Correlation': [], 
                       'knn_preservation': [], 
                       'ari': [],
                       'Correlation gt': [],
                       'knn_preservation gt': [],
                       'ari gt': [],
                       'Index': [], 
                       '% Corrupted': [], 
                       'Run_id': [], 
                       'grad_style': [], 
                       'method': [], 
                       'step': [], 
                       'removed_indices': []})

    #  Want this part to be deterministic!
    import os
    os.environ['PYTHONHASHSEED'] = str(0)
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    import random
    random.seed(0)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(0)

    tsne_output_dir = '{}/{}'.format(args.tsne_output_dir, args.run_id)
    # load attributions
    try:
        arr_obj = np.load(Path(tsne_output_dir) / 'tsne_results_style={}.npz'.format(args.grad_style), allow_pickle=True)
    except FileNotFoundError:
        print('Did not find {}. Running t-SNE now'.format(tsne_output_dir))
        # create attributions
        get_tsne_attributions(args.run_id, args.data_dir, args.tsne_output_dir, args.run_id, args.data_id, args.grad_style)
        arr_obj = np.load(Path(tsne_output_dir) / 'tsne_results_style={}.npz'.format(args.grad_style), allow_pickle=True)

    #  Load array object from tsne_output_dir
    out = arr_obj['out'].item()
    # Do distance computations once!
    dists_ref = scipy.spatial.distance.pdist(out['embeddings'][-1], metric='euclidean')
    X_reduced = get_data(args.data_id, args.data_dir)
    dists_real = scipy.spatial.distance.pdist(X_reduced, metric='euclidean')

    label_true = get_labels(args.data_id, args.data_dir)
    kmeans = KMeans(n_clusters=10, init='k-means++').fit(out['embeddings'][-1])
    label_tsne_orig = kmeans.labels_
    r_indices = np.random.choice(dists_ref.shape[0], size=20000, replace=False)
    print('ARI of ground truth versus t-SNE cluster labels for seed {}: {:.3f}'.format(args.run_id, 
                                                                                   adjusted_rand_score(label_true, label_tsne_orig)))
    print('KNN preservation of ground truth versus t-SNE for seed {}: {:.3f}'.format(args.run_id, 
                                                                                     compute_knn_preservation(dists_ref, dists_real, num_samples=label_true.shape[0], knn=10)))     
    print('Spearman correlation of ground truth versus t-SNE for seed {}: {:.3f}'.format(args.run_id, 
                                                                                     scipy.stats.spearmanr(dists_ref[r_indices], dists_real[r_indices]).correlation))

    if level == 'individual':
        fsscore, lscore, lscore_p_matrix, lscore_q_matrix = None, None, None, None
    elif level == 'class':
        # change label_true to label_tsne_orig to use t-SNE embedding based labels
        # not sure how this changes things. For now not using
        fsscore = None
        lscore = get_lap_score_features_per_class(X_reduced, label_true, arr_obj, args.step, add_diagonal=False, mode='default')
        lscore_q_matrix = get_lap_score_features_per_class(X_reduced, label_true, arr_obj, args.step, add_diagonal=False, mode='q_matrix')
        lscore_p_matrix = get_lap_score_features_per_class(X_reduced, label_true, arr_obj, args.step, add_diagonal=False, mode='p_matrix')
    elif level == 'global':
        fsscore = get_fisher_score_features(args.data_dir, args.data_id)
        lscore = get_lap_score_features(X_reduced, arr_obj, args.step, add_diagonal=False, mode='default')
        lscore_q_matrix = get_lap_score_features(X_reduced, arr_obj, args.step, add_diagonal=False, mode='q_matrix')
        lscore_p_matrix = get_lap_score_features(X_reduced, arr_obj, args.step, add_diagonal=False, mode='p_matrix')
    
    # repeat experiment corrupting up to 10 features
    num_feats = 50
    attrs = np.array(out['attrs'])[:,0,:,:]

    # clean up attributions
    attrs[np.isnan(attrs)] = 0
    attrs[attrs > 1] = 1
    attrs[attrs < -1] = -1
    
    for num_feats_corrupted in range(1, args.remove_until, 1):
        df = remove_each_step(args,
                              df,
                              num_feats,
                              num_feats_corrupted,
                              attrs,
                              tsne_output_dir, 
                              label_tsne_orig,
                              label_true,
                              lscore, 
                              lscore_p_matrix,
                              lscore_q_matrix,
                              fsscore,
                              dists_ref,
                              dists_real,
                              r_indices,
                              level)

def main(args):
    remove_all_steps(args, level='individual')




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
                        help='What methods of indices computation to use (does random by default): `top_attr`, `top_attr_ge_0`, `attr_feat`, `attr_ge_0_feat_ge_0`, `feat_size`')
    parser.add_argument('--final_csv_path', type=str,
                        help='where to save the final csv file')
    parser.add_argument('--remove_until', type=int, default=10,
                        help='Remove until this number of features')
    args = parser.parse_args()
    main(args)
