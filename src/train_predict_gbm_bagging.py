#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from sklearn.ensemble import BaggingClassifier as BG
from sklearn.ensemble import GradientBoostingClassifier as GBM

import argparse
import logging
import numpy as np
import time

import xgboost as xgb


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def train_predict(train_file, test_file, valid_train_file, valid_test_file,
                  predict_valid_file, predict_test_file,
                  n_est=100, depth=4, lrate=.1):

    logging.info('Loading training and test data...')
    X_valtrn, y_valtrn = load_svmlight_file(valid_train_file)
    X_valtst, y_valtst = load_svmlight_file(valid_test_file)

    logging.info('Validation...')
    gbm = GBM(max_depth=depth, learning_rate=lrate, n_estimators=n_est,
              random_state=2015)

    clf = BG(base_estimator=gbm, n_estimators=5, max_samples=0.8,
             max_features=0.8, bootstrap=True, bootstrap_features=True,
             random_state=42, verbose=0)

    clf.fit(X_valtrn.todense(), y_valtrn)
    p_valtst = clf.predict_proba(X_valtst.todense())[:, 1]
    lloss = log_loss(y_valtst, p_valtst)

    logging.info('Log Loss = {:.4f}'.format(lloss))

    logging.info('Retraining with 100% data...')
    X_trn, y_trn = load_svmlight_file(train_file)
    X_tst, _ = load_svmlight_file(test_file)

    clf.fit(X_trn.todense(), y_trn)
    p_tst = clf.predict_proba(X_tst.todense())[:, 1]

    logging.info('Saving predictions...')
    np.savetxt(predict_valid_file, p_valtst, fmt='%.6f')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--valid-train-file', required=True, dest='valid_train_file')
    parser.add_argument('--valid-test-file', required=True, dest='valid_test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--depth', type=int, dest='depth')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  valid_train_file=args.valid_train_file,
                  valid_test_file=args.valid_test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  depth=args.depth,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
