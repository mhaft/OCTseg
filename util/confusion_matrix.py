# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================
"""calculate confusion matrix. Confusion matrix contains

    * TP: True positive

    * TN: True negative

    * FP: False positive

    * FN: False Negative

    * TPR: True positive ratio or sensitivity

    * TNR: True Negative ratio or specificity

    * Dice Index

.. math::
    Dice = \\frac{
            2 \\times \\sum_{i \\in [[N]],  j \\in [[M]],  k \\in [[L]]} (label_{i,j,k} \  \\&\\& \  predict_{i,j,k})
        }{
            \\sum_{i \\in [[N]],  j \\in [[M]],  k \\in [[L]]} label_{i,j,k} \ + \
             \\sum_{i \\in [[N]],  j \\in [[M]],  k \\in [[L]]} predict_{i,j,k}
        }

Notes:
    Arguments are bash arguments.

Args:
    exp_def: experiment definition
    models_path: experiment definition
    epoch: model saved at this epoch
    useMask: use guide wire and nonIEL masks

Returns:
    add a line to the `../model/confusion_matrix.csv` file, which contains confusion matrix of testing and vali

"""

import argparse
import os

import tifffile
import h5py
import numpy as np


def confusion_matrix(label, target, mask):
    label, target = np.logical_and(label > 0, mask), np.logical_and(target > 0, mask)
    TP = np.sum(np.logical_and(label, target))
    TN = np.sum(np.logical_and(np.logical_not(label), np.logical_not(target))) - np.sum(np.logical_not(mask))
    FP = np.sum(np.logical_and(np.logical_not(label), target))
    FN = np.sum(np.logical_and(label, np.logical_not(target)))
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Dice = 2 * TP / (2 * TP + FP + FN)
    return TP, TN, FP, FN, TPR, TNR, Acc, Dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_def", type=str, default="1z", help="experiment definition")
    parser.add_argument("-models_path", type=str, default='../model/', help="experiment definition")
    parser.add_argument("-epoch", type=int, default=1000, help="model saved at this epoch")
    parser.add_argument("-useMask", type=int, default=0, help="use guide wire and nonIEL masks")
    args = parser.parse_args()

    label = tifffile.imread(args.models_path + args.exp_def + '/a-label.tif')
    target = tifffile.imread(args.models_path + args.exp_def + '/a-out-epoch%06d.tif' % args.epoch)
    log_file = args.models_path + args.exp_def + '/log-' + args.exp_def + '.csv'

    with open(log_file) as fp:
        line = fp.readline().split(',')
    for x in line:
        if x.startswith(" Namespace(data_path="):
            dataset_path = x[22:-1]
        elif x.startswith(" l="):
            L = int(x[3:])
        elif x.startswith(" w="):
            w = int(x[3:-2])
        elif x.startswith(" inCh="):
            inCh = int(x[6:])
        elif x.startswith(" nZ="):
            nZ = int(x[4:])
        elif x.startswith(" isCarts="):
            isCarts = int(x[9:])
            coord_sys = 'carts' if isCarts else 'polar'

    data_file = os.path.join(dataset_path, 'Dataset ' + coord_sys + ' Z%d-L%d-W%d-C%d.h5' % (nZ, L, w, inCh))
    report_file = '../model/confusion_matrix.csv'

    with h5py.File(data_file) as f:
        train_data_id = np.array(f.get('/train_data_id'))
        valid_data_id = np.array(f.get('/valid_data_id'))
        mask = np.array(f.get('/label')) > 0
        if len(mask.shape) > 4:
            mask = mask[:, mask.shape[1]//2, ...]
        if args.useMask:
            mask = np.logical_not(np.logical_or(mask[..., 0], mask[..., 1]))
        else:
            mask = np.ones(mask.shape[:-1])

        isTrain = []
        for i in np.union1d(train_data_id, valid_data_id):
            isTrain.append(True if i in train_data_id else False)

    train_confusion_matrix = confusion_matrix(label[isTrain, ...], target[isTrain, ...], mask[train_data_id, ...])
    valid_confusion_matrix = confusion_matrix(label[np.logical_not(isTrain), ...],
                                              target[np.logical_not(isTrain), ...], mask[valid_data_id, ...])

    if not os.path.exists(report_file):
        with open(report_file, 'w') as f:
            f.write('Model, Epoch, ' +
                    'Train TP, TN, FP, FN, TPR, TNR, Acc, Dice, ' +
                    'Valid TP, TN, FP, FN, TPR, TNR, Acc, Dice \n')

    with open(report_file, 'a') as f:
        f.write(('%s, %d' + 4 * ', %d' + 4 * ', %f' + 4 * ', %d' + 4 * ', %f' + '\n') % ((args.exp_def, args.epoch, ) +
                train_confusion_matrix + valid_confusion_matrix))

    print('Summ.\t' + 4 * '%s\t' % ('TPR', 'TNR', 'Acc', 'Dice'))
    print('Train\t' + 4 * '%.2f\t' % train_confusion_matrix[-4:])
    print('Valid\t' + 4 * '%.2f\t' % valid_confusion_matrix[-4:])
    print('Saved in %s' % report_file)
