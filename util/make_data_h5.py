# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Convert an 2D or 3D image from polar or cylindrical coordinate to the
    cartesian coordinate."""

from __future__ import absolute_import, division, print_function

import glob
import tifffile

import numpy as np


def im_fix_width(im, w):
    """pad or crop the 3D image to have width and length equal to the input width"""
    w0 = im.shape[1]
    if w <= w0:
        i_start = (w0 - w) // 2
        im = im[:, i_start:(i_start + w), i_start:(i_start + w)]
    else:
        pad1 = (w - w0) // 2
        pad2 = w - w0 - pad1
        im = np.pad(im, ((0, 0), (pad1, pad2), (pad1, pad2)), 'constant', constant_values=0)
    return im


def make_data_h5(folder_path, im_shape):
    if len(im_shape) == 3:
        im_shape = tuple(im_shape) + (1, )
    tmp_im = np.zeros((1,) + tuple(im_shape), dtype='uint8')
    tmp_label = np.zeros((1, 1,) + tuple(im_shape[1:-1]) + (1, ), dtype='uint8')
    im = np.zeros((0,)+tuple(im_shape), dtype='uint8')
    label = np.zeros((0, 1,) + tuple(im_shape[1:-1]) + (1,), dtype='uint8')
    cases = glob.glob(folder_path + '*-Seg.tif')
    for case in cases:
        tmp = tifffile.imread(case)
        tmp = im_fix_width(tmp, im_shape[1])
        slice_list = np.nonzero(np.any(np.any(tmp, axis=-1), axis=-1))[0]
        for i in slice_list:
            tmp_label[0, 0, :, :, 0] = tmp[i, ...]
            label = np.concatenate((label, tmp_label), axis=0)
        tmp = tifffile.imread(case[:-8] + '*-im.tif')
        tmp = im_fix_width(tmp, im_shape[1])
        if im_shape[-1] == 1:
            tmp = np.expand_dims(tmp[1::3, ...], axis=-1)
        else:
            tmp = np.moveaxis(np.reshape(tmp, (-1, 3,) + tmp.shape[1:]), 1, -1)
        for i in slice_list:
            tmp_im[:] = tmp[(i - (im_shape[0] // 2)):(i - (im_shape[0]//2) + im_shape[0]), ...]
            im = np.concatenate((im, tmp_im), axis=0)
    return im, label
