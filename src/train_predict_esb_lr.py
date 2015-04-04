#!/usr/bin/env python

import argparse
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

from kaggler.logger import log
from kaggler.util import load_data


def train_predict_lr_cv(train_file, test_file, predict_train_file,
                        predict_test_file, c, n_fold=10):
    log.info("Reading in the training data")
    X_trn, y_trn = load_data(train_file, dense=True)

    log.info("Reading in the test data")
    X_tst, _ = load_data(test_file, dense=True)

    log.info('Normalizing data')
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_trn)
    X_tst = scaler.transform(X_tst)
 
    cv = cross_validation.StratifiedKFold(y_trn, n_folds=n_fold, shuffle=True,
                                          random_state=1)

    yhat_tst = np.zeros((X_tst.shape[0], ))
    yhat_trn = np.zeros((X_trn.shape[0], ))
    for i, (i_trn, i_val) in enumerate(cv, start=1):
        log.info('Training CV #{}'.format(i))
        clf = LogisticRegression(C=c, class_weight='auto', random_state=2013)
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
    parser.add_argument('--c', '-c', required=True, type=float, dest='c')

    args = parser.parse_args()

    start = time.time()
    train_predict_lr_cv(train_file=args.train,
                        test_file=args.test,
                        predict_train_file=args.predict_train,
                        predict_test_file=args.predict_test,
                        c=args.c)

    log.info('Finished ({:.2f} sec elasped).'.format(time.time() - start))
