#!/usr/bin/env python

import argparse
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.ensemble import GradientBoostingClassifier
import time

from kaggler.const import FIXED_SEED, SEC_PER_MIN
from kaggler.logger import log
from kaggler.util import load_data


def train_predict_esb_gbm(train_file, test_file, predict_train_file,
                        predict_test_file, n_est, l_rate, m_depth, n_fold=10):
    log.info("Reading in the training data")
    X_trn, y_trn = load_data(train_file, dense=True)

    log.info("Reading in the test data")
    X_tst, _ = load_data(test_file, dense=True)

    cv = cross_validation.StratifiedKFold(y_trn, n_folds=n_fold, shuffle=True,
                                          random_state=1)

    yhat_tst = np.zeros((X_tst.shape[0], ))
    yhat_trn = np.zeros((X_trn.shape[0], ))
    for i, (i_trn, i_val) in enumerate(cv, start=1):
        log.info('Training CV #{}'.format(i))
        clf = GradientBoostingClassifier(n_estimators=n_est,
                                         learning_rate=l_rate,
                                         max_depth=m_depth,
                                         min_samples_split=10,
                                         verbose=1,
                                         random_state=FIXED_SEED)
     
        clf.fit(X_trn[i_trn], y_trn[i_trn])

        yhat_trn[i_val] = clf.predict_proba(X_trn[i_val])[:, 1]
        yhat_tst += np.array(clf.predict_proba(X_tst)[:, 1]) / n_fold

    auc_cv = metrics.roc_auc_score(y_trn, yhat_trn)
    log.info('AUC CV: {}'.format(auc_cv))
    log.info("Writing test predictions to file")
    np.savetxt(predict_train_file, yhat_trn, fmt='%.6f', delimiter=',')
    np.savetxt(predict_test_file, yhat_tst, fmt='%.6f', delimiter=',')

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', '-t', required=True, dest='train')
    parser.add_argument('--test-file', '-v', required=True, dest='test')
    parser.add_argument('--predict-train-file', '-p', required=True,
                        dest='predict_train')
    parser.add_argument('--predict-test-file', '-q', required=True,
                        dest='predict_test')
    parser.add_argument('--l_rate', '-l', required=True, type=float,
                        dest='l_rate')
    parser.add_argument('--m_depth', '-m', required=True, type=int,
                        dest='m_depth')
    parser.add_argument('--n_est', '-n', required=True, type=int, dest='n_est')

    args = parser.parse_args()

    start = time.time()
    train_predict_esb_gbm(train_file=args.train,
                          test_file=args.test,
                          predict_train_file=args.predict_train,
                          predict_test_file=args.predict_test,
                          l_rate=args.l_rate,
                          m_depth=args.m_depth,
                          n_est=args.n_est)

    log.info('finished ({:.2f} min elasped).'.format((time.time() - start) /
                                                     SEC_PER_MIN))
