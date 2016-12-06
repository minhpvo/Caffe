import numpy as np
import os.path as osp
from argparse import ArgumentParser


def main(args):
    with open(args.gt_file) as f:
        lines = f.readlines()
        gt = [int(line.strip().split()[1]) for line in lines]
        gt = np.asarray(gt).reshape(len(gt), 1)
    feat = np.load(args.feat_files[0])
    for f in args.feat_files[1:]:
        feat = np.vstack((feat, np.load(f)))
    pred = np.asarray(np.argsort(feat[:len(gt)], axis=1)[:, ::-1])
    for k in [1, 5]:
        acc = (pred[:, :k] == gt).sum(axis=1).mean()
        print 'Top-{} accuracy: {:.2%}'.format(k, acc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('gt_file')
    parser.add_argument('--feat_files', nargs='+')
    args = parser.parse_args()
    main(args)