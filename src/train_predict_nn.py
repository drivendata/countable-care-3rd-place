#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss

import argparse
import logging
import numpy as np
import time

from kaggler.online_model import NN


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_iter=100, hidden=4, lrate=.1, n_fold=5):

    _, y_trn = load_svmlight_file(train_file)

    cv = StratifiedKFold(y_trn, n_folds=n_fold, shuffle=True, random_state=2015)

    logging.info('Cross validation...')
    p_val = np.zeros_like(y_trn)
    lloss = 0.
    for i_trn, i_val in cv:
        clf = NN(n=5200, h=hidden, a=lrate, seed=2015)
        val = []
        # train for cross validation
        for i_iter in range(n_iter):
            for i, (x, y) in enumerate(clf.read_sparse(train_file)):
                if i in i_val:
                    if i_iter == 0:
                        val.append((i, x, y))
                else:
                    p = clf.predict(x)
                    clf.update(x, p - y)

            # predict for cross validation
            for i, x, y in val:
                p_val[i] = clf.predict(x)

            logging.info('Epoch #{}: Log Loss = {:.4f}'.format(i_iter + 1,
                                                               log_loss(y_trn[i_val], p_val[i_val])))

        lloss += log_loss(y_trn[i_val], p_val[i_val])

    logging.info('Log Loss = {:.4f}'.format(lloss / n_fold))

    logging.info('Retraining with 100% data...')
    clf = NN(n=5200, h=hidden, a=lrate, seed=2015)
    for i_iter in range(n_iter):
        for x, y in clf.read_sparse(train_file):
            p = clf.predict(x)
            clf.update(x, p - y)

        logging.info('Epoch #{}'.format(i_iter + 1))

    _, y_tst = load_svmlight_file(test_file)
    p_tst = np.zeros_like(y_tst)
    for i, (x, _) in enumerate(clf.read_sparse(test_file)):
        p_tst[i] = clf.predict(x)

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
    parser.add_argument('--hidden', type=int, dest='hidden')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_iter=args.n_iter,
                  hidden=args.hidden,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
