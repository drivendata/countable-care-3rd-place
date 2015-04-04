#!/usr/bin/env python

from itertools import izip
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-file', required=True, dest='feature')
    parser.add_argument('--train-file', required=True, dest='train')
    parser.add_argument('--valid-file', required=True, dest='valid')
    parser.add_argument('--valid-id', required=True, dest='valid_id')

    args = parser.parse_args()

    with open(args.valid_id) as f_val_id, open(args.feature) as f_feature, \
         open(args.train, 'w') as f_trn, open(args.valid, 'w') as f_val:
        for is_valid, row in izip(f_val_id, f_feature):
            is_valid = int(is_valid.strip())
            if is_valid == 0:
                f_trn.write(row)
            else:
                f_val.write(row)

