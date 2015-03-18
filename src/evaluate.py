#!/usr/bin/env python

from __future__ import division
from sklearn.metrics import log_loss

import argparse
import numpy as np
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-file', '-t', required=True, dest='target_file')
    parser.add_argument('--predict-file', '-p', required=True, dest='predict_file')
    args = parser.parse_args()

    p = np.loadtxt(args.predict_file, delimiter=',')
    y = np.loadtxt(args.target_file, delimiter=',')

    model_name = os.path.splitext(args.predict_file)[0]

    lloss = 0.
    for i in range(p.shape[1]):
        lloss += log_loss(y[:, i], p[:, i])

    print('{}\t{:.6f}'.format(model_name, lloss / p.shape[1]))
