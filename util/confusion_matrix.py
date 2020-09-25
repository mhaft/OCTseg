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
import time

import tifffile
import h5py
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from util.read_parameter_from_log_file import read_parameter_from_log_file


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


def boundary_accuracy(label, target):

    def boundary_mask(im_2d):
        im_2d = np.pad(im_2d, ((0, 1), (0, 1)), mode='edge')
        return distance_transform_edt(np.logical_and(im_2d[:-1, :-1] == im_2d[1:, :-1],
                                                    im_2d[:-1, :-1] == im_2d[:-1, 1:]).astype('float'))

    def boundary_error_2d(label, target):
        label, target = boundary_mask(label), boundary_mask(target)

        return np.concatenate(([np.max([np.mean(label[target == 0]), np.mean(target[label == 0])])],
                               np.percentile(label[target == 0], [50, 90, 95, 100]),
                               [np.mean(label[target == 0])])) \
            if np.any(target == 0) else np.zeros(6) + np.inf

    out = np.zeros(5)
    for i in range(label.shape[0]):
        out += boundary_error_2d(label[i, ...], target[i, ...])
    out /= label.shape[0]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_def", type=str, default="1z", help="experiment definition")
    parser.add_argument("-models_path", type=str, default='../model/', help="experiment definition")
    parser.add_argument("-testEpoch", type=int, default=1000, help="model saved at this epoch")
    parser.add_argument("-useMask", type=int, default=0, help="use guide wire and nonIEL masks")
    args = parser.parse_args()

    log_file = os.path.join(args.models_path, args.exp_def, 'log-' + args.exp_def + '.csv')
    args = read_parameter_from_log_file(args, log_file)
    coord_sys = 'carts' if args.isCarts else 'polar'

    label = np.mod(tifffile.imread(args.models_path + args.exp_def + '/a-label.tif'), args.outCh)
    target = tifffile.imread(args.models_path + args.exp_def + '/a-out-epoch%06d.tif' % args.testEpoch)

    data_file = os.path.join(args.data_path, 'Dataset ' + coord_sys + ' Z%d-L%d-W%d-C%d.h5' % (args.nZ, args.l, args.w,
                                                                                               args.inCh))
    if not os.path.exists(data_file):
        args.data_path = "D:\\MLIntravascularPolarimetry\\MLCardioPullbacks\\"
        data_file = os.path.join(args.data_path, 'Dataset ' + coord_sys + ' Z%d-L%d-W%d-C%d.h5' %
                                 (args.nZ, args.l, args.w, args.inCh))

    report_file = '../model/confusion_matrix.csv'

    with h5py.File(data_file, 'r') as f:
        train_data_id = np.array(f.get('/train_data_id'))
        valid_data_id = np.array(f.get('/valid_data_id'))
        train_valid_data_id = np.union1d(train_data_id, valid_data_id)
        if args.useMask:
            mask = np.array(f.get('/label'))[train_valid_data_id, ...] > 0
            if len(mask.shape) > 4:
                mask = mask[:, mask.shape[1]//2, ...]
            mask = np.logical_not(np.logical_or(mask[..., 0], mask[..., 1]))
        else:
            mask = np.ones((train_valid_data_id.size,) + (1,) * (label.ndim - 1))

    isTrain = []
    for i in train_valid_data_id:
        isTrain.append(True if i in train_data_id else False)

    classes = [1] if args.outCh == 2 else range(args.outCh)
    if not os.path.exists(report_file):
        with open(report_file, 'w') as f:
            f.write('Model, Epoch, ' +
                    (('Class %d Train TP, TN, FP, FN, TPR, TNR, Acc, Dice, ' +
                     'Valid TP, TN, FP, FN, TPR, TNR, Acc, Dice, ') * len(classes)) % tuple(classes) +
                     'Boundary MHD, Median, 90%, 95%, Max, Avg Err' + '\n')

    report_out = '%s, %d' % (args.exp_def, args.testEpoch,)
    for i_class in classes:
        train_confusion_matrix = confusion_matrix(label[isTrain, ...] == i_class,
                                                  target[isTrain, ...] == i_class, mask[isTrain, ...])
        valid_confusion_matrix = confusion_matrix(label[np.logical_not(isTrain), ...] == i_class,
                                                  target[np.logical_not(isTrain), ...] == i_class,
                                                  mask[np.logical_not(isTrain), ...])

        report_out += (4 * ', %d' + 4 * ', %f' + 4 * ', %d' + 4 * ', %f') % (train_confusion_matrix +
                                                                             valid_confusion_matrix)
        print('Summ.\t' + 4 * '%s\t' % ('TPR', 'TNR', 'Acc', 'Dice') + '\tClass %d' % i_class)
        print('Train\t' + 4 * '%.2f\t' % train_confusion_matrix[-4:])
        print('Valid\t' + 4 * '%.2f\t' % valid_confusion_matrix[-4:])

    train_boundary_accuracy = boundary_accuracy(label[isTrain, ...], target[isTrain, ...])
    valid_boundary_accuracy = boundary_accuracy(label[np.logical_not(isTrain), ...],
                                                target[np.logical_not(isTrain), ...])

    report_out += (12 * ', %f') % (tuple(train_boundary_accuracy) + tuple(valid_boundary_accuracy))

    print('Boundary Accuracy')
    print('Summ.\t' + 6 * '%s\t' % ('MHD', '50%', '90%', '95%', 'Max', 'Avg Err'))
    print('Train\t' + 6 * '%.2f\t' % tuple(train_boundary_accuracy))
    print('Valid\t' + 6 * '%.2f\t' % tuple(valid_boundary_accuracy))

    report_out += '\n'
    try:
        with open(report_file, 'a') as f:
            f.write(report_out)
    except IOError:
        time.sleep(3)
        with open(report_file, 'a') as f:
            f.write(report_out)

    print('Saved in %s' % report_file)
