# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Read ROI file generated based on the and generate segmentation results."""

from __future__ import absolute_import, division, print_function

import csv
import numpy as np

from scipy.interpolate import interp1d


def roi_file_parser(file_path):
    """Parse roi file and output the lists of objects"""
    obj_list, last_row, last_case = {'Lumen':[], 'IEL':[], 'GW':[]}, [''], ''
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            case = row[0]
            if case in ['ROIformat', 'closed']:
                last_case = ''
            if case in ['Snake', 'Angle']:
                if case != last_row[0]:
                    last_case = row[-1]
                    obj_list[last_case].append([])
                obj_list[last_case][-1].append([int(i) for i in row[1:4]])
            last_row = row
    return obj_list


def lumen_iel_mask(obj_list, im_shape):
    """generate lumen or IEL mask based on the point list."""
    obj_list = np.array(obj_list) - 1 # match 1-index to 0-index
    x, idx, c = np.unique(obj_list[:, 1], return_index=True, return_counts=True)
    obj_list = obj_list[idx[c < 2]]
    obj_list = np.concatenate((obj_list - [0, im_shape[-1], 0], obj_list, obj_list +
                               [0, im_shape[-1], 0]), axis=0)
    f = interp1d(obj_list[:, 1], obj_list[:, 0], kind='quadratic')
    y_lim = np.tile(f(np.arange(im_shape[-1])).T, reps=(im_shape[-2], 1))
    y = np.tile(np.arange(im_shape[-2]).T, reps=(im_shape[-1], 1)).T
    return y <= y_lim


def read_oct_roi_file(file_path, im_shape):
    obj_list = roi_file_parser(file_path)
    out = np.zeros(im_shape, dtype='uint8')
    for iel in obj_list['IEL']:
        tmp = out[iel[0][2] - 1, ...]
        tmp[lumen_iel_mask(iel, im_shape)] = 3
        out[iel[0][2] - 1, ...] = tmp
    for lumen in obj_list['Lumen']:
        tmp = out[lumen[0][2] - 1, ...]
        tmp[lumen_iel_mask(lumen, im_shape)] = 2
        out[lumen[0][2] - 1, ...] = tmp
    for gw in obj_list['GW']:
        z = gw[1][2] - 1
        gw = np.array([gw[1][1], gw[2][1]])
        if gw[0] <= gw[1]:
            out[z, :, gw[0]:(gw[1] + 1)] = 1
        else:
            out[z, :, gw[0]:] = 1
            out[z, :, :(gw[1] + 1)] = 1
    return out

