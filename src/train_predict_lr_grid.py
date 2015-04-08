#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression as LR

import argparse
import logging
import numpy as np
import os
import time


def train_predict(train_file, test_file, valid_train_file, valid_test_file,
                  predict_valid_file, predict_test_file):

    feature_name = os.path.basename(train_file)[:-10]
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='lr_grid_{}.log'.format(feature_name))

    logging.info('Loading training and test data...')
    X_valtrn, y_valtrn = load_svmlight_file(valid_train_file)
    X_valtst, y_valtst = load_svmlight_file(valid_test_file)

    lr = LR(penalty='l2', dual=True, class_weight='auto',
            random_state=2015)
    param = {'C': [.01, .05, .1, .5, 1]}
    clf = GridSearchCV(lr, param, scoring='log_loss', verbose=1)

    logging.info('Cross validation for grid search...')
    clf.fit(X_valtrn, y_valtrn)

    logging.info('best score = {:.4f}'.format(clf.best_score_))
    logging.info('best model = {}'.format(clf.best_estimator_))

    clf.best_estimator_.fit(X_valtrn, y_valtrn)
    p_valtst = clf.best_estimator_.predict_proba(X_valtst)[:, 1]
    lloss = log_loss(y_valtst, p_valtst)

    logging.info('Log Loss = {:.4f}'.format(lloss))

    logging.info('Retraining with 100% data...')
    X, y = load_svmlight_file(train_file)
    X_tst, _ = load_svmlight_file(test_file)

    clf.best_estimator_.fit(X, y)
    p_tst = clf.best_estimator_.predict_proba(X_tst)[:, 1]

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

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  valid_train_file=args.valid_train_file,
                  valid_test_file=args.valid_test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
