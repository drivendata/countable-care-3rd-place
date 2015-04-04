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


def train_predict(train_file, test_file, model_file, valid_train_file,
                  valid_test_file, valid_model_file,
                  predict_valid_file, predict_test_file,
                  n_iter=100, dim=4, lrate=.1):

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename='ffm_{}_{}_{}.log'.format(
                                                        n_iter, dim, lrate
                                                      ))

    logging.info('Validation...')
    subprocess.call(["ffm-train",
                     '-k', str(dim),
                     '-r', str(lrate),
                     '-t', str(n_iter),
                     '-p', valid_test_file,
                     valid_train_file,
                     valid_model_file])

    subprocess.call(["ffm-predict",
                     valid_test_file,
                     valid_model_file,
                     predict_valid_file])

    logging.info('Retraining with 100% data...')
    subprocess.call(["ffm-train",
                     '-k', str(dim),
                     '-r', str(lrate),
                     '-t', str(n_iter),
                     train_file,
                     model_file])

    subprocess.call(["ffm-predict",
                     test_file,
                     model_file,
                     predict_test_file])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--model-file', required=True, dest='model_file')
    parser.add_argument('--valid-train-file', required=True, dest='valid_train_file')
    parser.add_argument('--valid-test-file', required=True, dest='valid_test_file')
    parser.add_argument('--valid-model-file', required=True, dest='valid_model_file')
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
                  model_file=args.model_file,
                  valid_train_file=args.valid_train_file,
                  valid_test_file=args.valid_test_file,
                  valid_model_file=args.valid_model_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_iter=args.n_iter,
                  dim=args.dim,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
