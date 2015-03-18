#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression as LR

import argparse
import logging
import numpy as np
import time
import xgboost as xgb


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)

param = {'objective': 'binary:logistic',
         'eval_metric': 'logloss',
         'nthread': 4}


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_iter=100, depth=4, lrate=.1, n_fold=10):

    logging.info('Loading training and test data...')
    X, y = load_svmlight_file(train_file)
    X_tst, _ = load_svmlight_file(test_file)

    X = X.todense()
    X_tst = X_tst.todense()

    param['num_round'] = n_iter
    param['max_depth'] = depth
    param['eta'] = lrate

    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)

    logging.info('Cross validation...')
    p_val = np.zeros_like(y)
    lloss = 0.
    for i_trn, i_val in cv:
        dtrain = xgb.DMatrix(X[i_trn], label=y[i_trn])
        dtest = xgb.DMatrix(X[i_val], label=y[i_val])

        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        clf = xgb.train(param.items(), dtrain, n_iter, evallist)
        p_val[i_val] = clf.predict(dtest)
        lloss += log_loss(y[i_val], p_val[i_val])

    logging.info('Log Loss = {:.4f}'.format(lloss / n_fold))

    logging.info('Retraining with 100% data...')
    clf = xgb.train(param.items(), xgb.DMatrix(X, label=y), n_iter, evallist)
    p_tst = clf.predict(xgb.DMatrix(X_tst))

    logging.info('Saving predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-iter', type=int, dest='n_iter')
    parser.add_argument('--depth', type=int, dest='depth')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_iter=args.n_iter,
                  depth=args.depth,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
