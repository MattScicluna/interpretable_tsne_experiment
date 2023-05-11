import os
#import sys
from pathlib import Path
#import pickle
import argparse
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#import gensim
from gensim import models
#from gensim.models import KeyedVectors
#from gensim.parsing.preprocessing import remove_stopwords
#from gensim import corpora
#from gensim.utils import lemmatize
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


#  Helper funcs taken from:
#  https://towardsdatascience.com/using-word2vec-to-analyze-news-headlines-and-predict-article-success-cdeda5f14751
def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)


# Our earlier preprocessing was done when we were dealing only with word vectors
# Here, we need each document to remain a document
def preprocess(text, stop_words):
    #text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    return doc


# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)


def sentence_in_threshold(word2vec_model, doc, min_thresh=25, max_thresh=750):
    return min_thresh <= len([word in word2vec_model.vocab for word in doc]) <= max_thresh


# Filter out documents
def filter_docs(corpus, texts, targets, condition):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc, c) in zip(texts, corpus, condition)
                 if c]

    corpus = [doc for (doc, c) in zip(corpus, condition) if c]
    targets = [t for (t, c) in zip(targets, condition) if c]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, targets)


def make_mapping(raw_sentence, word_list):
    sent = list(filter(None, raw_sentence.split('\n')))  # Note: this mutates raw_sent (X)!
    shape = (len(sent),
             max([len(sent[i]) for i in range(len(sent))]))

    attr_mapping = np.ma.masked_array(data=np.zeros(shape, dtype=int),
                                      mask=True)

    for i, token in enumerate(word_list):
        for j, s in enumerate(sent):
            for (s,e) in [m.span() for m in re.finditer(re.compile(r'\b{}\b'.format(token)), s)]:
                attr_mapping[j, s:e] = np.array([i]*(e-s))

    null_val = attr_mapping.max()+1
    attr_mapping.set_fill_value(null_val)
    return attr_mapping


def create_20ng(args):

    data_dir = Path(args.data_dir)

    # Load raw data from https://www.openml.org/d/554
    print('loading 20NG data from {}'.format(data_dir))
    try:
        _path = '{}/{}/processed_data.npz'.format(data_dir, args.id)
        files = np.load(_path, allow_pickle=True)
        files['X_reduced']
        files['X_words']
        files['labels']
        files['attr_mapping']
        print('Found processed data at {}. Skipping...'.format(_path))
    except:
        nltk.data.path.append('{}/nltk'.format(args.data_dir))
        nltk.download('punkt', download_dir='{}/nltk'.format(args.data_dir))
        nltk.download('stopwords', download_dir='{}/nltk'.format(args.data_dir))

        stop_words = set(stopwords.words('english'))

        # Load raw data from https://www.openml.org/d/554
        print('loading 20 NewsGroup data from {}'.format(data_dir))
        twenty_train = fetch_20newsgroups(subset='all',
                                          remove=('headers', 'footers', 'quotes'),
                                          shuffle=True,
                                          random_state=args.seed,
                                          data_home=args.data_dir)

        X = twenty_train.data
        y = twenty_train.target

        # TODO: remove these lines!
        #X = [X[i] for i in range(5000)]
        #y = y[:5000]

        y_train = pd.Categorical(y)
        y_train = y_train.rename_categories({i: t for i,t in enumerate(twenty_train.target_names)})

        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))  # map punctuation to space
        #raw_sentence = raw_sentence.translate(translator).lower()
        X = [X[i].translate(translator) for i in range(len(X))]

        # processed data will be put into a new directory
        data_dir = data_dir / str(args.id)
        os.makedirs(data_dir, exist_ok=True)
        random_state = check_random_state(args.seed)
        model = models.KeyedVectors.load_word2vec_format(args.model_dir, binary=True)

        sent_list = X
        del X
        # Preprocess the corpus
        corpus = [preprocess(sent, stop_words) for sent in sent_list]
        #condition = [has_vector_representation(model, doc) for doc in corpus]
        condition = [sentence_in_threshold(model,
                                           doc,
                                           args.min_word_thresh,
                                           args.max_word_thresh) for doc in corpus]

        # Remove docs that don't include any words in W2V's vocab
        corpus, sent_list, y_train = filter_docs(corpus, sent_list, y_train, condition)

        # Filter out any empty docs
        condition = [(len(doc) != 0) for doc in corpus]
        corpus, sent_list, y_train = filter_docs(corpus, sent_list, y_train, condition)

        x = []
        for doc in corpus:  # append the vector for each document
            x.append(document_vector(model, doc))

        X_reduced = np.array(x)  # list to array

        print('Dataset dimension: {}'.format(X_reduced.shape))

        indices = np.arange(len(corpus))
        # select subset of indices <= args.size
        permutation = random_state.permutation(indices)
        indices = permutation[:args.size]

        X_reduced = X_reduced[indices]
        #  only gets unique tokens and sorts them alphabetically!
        corpus = [np.unique([word for word in corpus[i] if word in model.vocab]) for i in indices]
        sent_list = [sent_list[i].translate(translator) for i in indices]  # .lower()
        y_train = [y_train[i] for i in indices]
        print('created subsets')

        # maps positions in raw sentences to index of word in corpus
        attr_mapping = [make_mapping(sent_list[i], corpus[i]) for i in range(len(corpus))]
        print('created mappings')

        #reduce dimensionality to 50 before running tsne
        if args.num_pca != 0:
            pca = PCA(n_components=args.num_pca, random_state=args.seed)
            X_reduced = pca.fit_transform(X_reduced)
            pca_comp = pca.components_
            pca_mean = None  # not needed for inverse transform
            print('reduced dimension to {} using pca'.format(args.num_pca))
        else:
            X_reduced = X_reduced
            pca_comp = None
            pca_mean = None
            pca = None

        np.savez(data_dir / 'processed_data',
                 X_reduced=np.array(X_reduced, dtype=np.double),
                 X_words=corpus,
                 labels=y_train,
                 attr_mapping=attr_mapping,
                 sent_list=sent_list,
                 permutation=permutation,
                 seed=args.seed,
                 pca_comp=pca_comp,
                 pca_mean=pca_mean,
                 pca_obj=pca,
                 model_dir=args.model_dir
                 )

    print('saved processed data to {}'.format(data_dir / 'processed_data'))


def main():
    parser = argparse.ArgumentParser(description='Create 20ng (cleaned) dataset')
    parser.add_argument('--data_dir', type=str,
                        help='directory of where to save/load 20ng data from')
    parser.add_argument('--size', type=int, default=10000,
                        help='how large should the dataset be?')
    parser.add_argument('--seed', type=int, default=8,
                        help='random seed (for subsetting, etc...)')
    parser.add_argument('--id', type=int, default=0,
                        help='id of 20ng processing')
    parser.add_argument('--num_pca', type=int, default=50,
                        help='number of pcas to keep for 20ng processing (0 = no pca)')
    parser.add_argument('--min_word_thresh', type=int, default=25,
                        help='keep sentences with >= this')
    parser.add_argument('--max_word_thresh', type=int, default=750,
                        help='keep sentences with <= this')
    parser.add_argument('--model_dir', type=str,
                        help='directory of where to load word2vec model from')

    args = parser.parse_args()
    create_20ng(args)


if __name__ == '__main__':
    main()
