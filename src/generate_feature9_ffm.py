#!/usr/bin/env python

from scipy import sparse
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import argparse
import logging
import numpy as np
import os
import pandas as pd

from kaggler.util import get_label_encoder
from kaggler.util import encode_categorical_features, normalize_numerical_feature


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def generate_feature(train_file, label_file, test_file, feature_dir, name):
    # Load data files
    logging.info('Loading training and test data')
    trn = pd.read_csv(train_file, index_col=0)
    tst = pd.read_csv(test_file, index_col=0)
    label = pd.read_csv(label_file, index_col=0)
    n_trn = trn.shape[0]
    n_tst = tst.shape[0]

    logging.info('Combining training and test data')
    df = pd.concat([trn, tst], ignore_index=True)

    cols = list(df.columns)
    num_cols = [x for x in cols if x[0] == 'n']
    ord_cols = [x for x in cols if x[0] == 'o']
    cat_cols = [x for x in cols if x[0] == 'c' or x[0] == 'r']

    # no transformation for numerical variables
    logging.info('Imputing missing values in numerical columns by 0')

    # log2(1 + x) transformation for count variables
    for col in num_cols:
        df[col] = df[col].apply(lambda x: int(x * 100) if pd.notnull(x) else x)

    for col in ord_cols:
        df[col] = df[col].apply(lambda x: int(np.log2(1 + x)) if pd.notnull(x) else x)

    # One-Hot-Encoding for categorical variables
    logging.info('One-hot-encoding categorical columns')

    for i, col in enumerate(cols, start=1):
        label_encoder = get_label_encoder(df.ix[:n_trn, col], min_obs=3)
        feature = df[col].apply(lambda x: label_encoder.get(x, 0))
        feature += 1
        feature[feature < 0] = 0

        df[col] = feature.apply(lambda x: ' {}:{}:1'.format(i, x) if x > 0 else '')
        logging.info('{} processed'.format(col))

    logging.info('Saving features into {}'.format(feature_dir))
    for i in range(label.shape[1]):
        f_trn = open(os.path.join(feature_dir,
                                  '{}.trn{:02d}.ffm'.format(name, i)), 'w')
        f_tst = open(os.path.join(feature_dir,
                                  '{}.tst{:02d}.ffm'.format(name, i)), 'w')

        for j in range(df.shape[0]):
            features = ''.join(df.ix[j, :].tolist()).strip()
            
            if j < n_trn:
                f_trn.write('{} {}\n'.format(int(label.iloc[j, i]), features))
            else:
                f_tst.write('0 {}\n'.format(features))

        f_trn.close()
        f_tst.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train')
    parser.add_argument('--test-file', required=True, dest='test')
    parser.add_argument('--label-file', required=True, dest='label')
    parser.add_argument('--feature-dir', required=True, dest='feature_dir')
    parser.add_argument('--feature-name', required=True, dest='feature_name')

    args = parser.parse_args()

    generate_feature(train_file=args.train,
                     label_file=args.label,
                     test_file=args.test,
                     feature_dir=args.feature_dir,
                     name=args.feature_name)
