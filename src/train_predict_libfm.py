#!/usr/bin/env python

from __future__ import division
from datetime import datetime
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import log_loss

import argparse
import logging
import numpy as np
import os
import subprocess
import time


def train_predict(train_file, test_file, valid_train_file, valid_test_file,
                  predict_valid_file, predict_test_file,
                  n_iter=100, dim=4, lrate=.1, n_fold=5):

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename='libfm_{}_{}_{}.log'.format(
                                                        n_iter, dim, lrate
                                                      ))

    logging.info('Loading training data')
    _, y = load_svmlight_file(valid_test_file)

    subprocess.call(["libFM",
                     "-task", "c",
                     '-dim', '1,1,{}'.format(dim),
                     '-init_stdev', str(lrate),
                     '-iter', str(n_iter),
                     '-train', valid_train_file,
                     '-test', valid_test_file,
                     '-out', predict_valid_file])

    p = np.loadtxt(predict_valid_file)
    lloss = log_loss(y, p)

    logging.info('Log Loss = {:.4f}'.format(lloss))

    logging.info('Retraining with 100% data...')
    subprocess.call(["libFM",
                     "-task", "c",
                     '-dim', '1,1,{}'.format(dim),
                     '-init_stdev', str(lrate),
                     '-iter', str(n_iter),
                     '-train', train_file,
                     '-test', test_file,
                     '-out', predict_test_file])


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
    parser.add_argument('--n-iter', type=int, dest='n_iter')
    parser.add_argument('--dim', type=int, dest='dim')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  valid_train_file=args.valid_train_file,
                  valid_test_file=args.valid_test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_iter=args.n_iter,
                  dim=args.dim,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
