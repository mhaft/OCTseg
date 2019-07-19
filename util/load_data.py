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
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom


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


def make_dataset(folder_path, im_shape):
    if len(im_shape) == 3:
        im_shape = tuple(im_shape) + (1, )
    tmp_im = np.zeros((1,) + tuple(im_shape), dtype='uint8')
    tmp_label = np.zeros((1,) + tuple(im_shape[:-1]) + (1, ), dtype='uint8')
    im = np.zeros((0,)+tuple(im_shape), dtype='uint8')
    label = np.zeros((0, ) + tuple(im_shape[:-1]) + (1,), dtype='uint8')
    cases = glob.glob(folder_path + '*-Seg.tif')
    z_pad = (im_shape[0] - 1) // 2
    sample_caseID = []
    for i_case in tqdm(range(len(cases))):
        case = cases[i_case]
        tmp = tifffile.imread(case)
        tmp = im_fix_width(tmp, 512)
        slice_list = np.nonzero(np.any(np.any(tmp > 1, axis=-1), axis=-1))[0]
        for i in slice_list:
            j1, j2, j3, j4 = max(0, i - z_pad), i + z_pad + 1, max(0, z_pad - i), max(0, z_pad - i) + im_shape[0]
            tmp_label[0, j3:j4, :, :, 0] = zoom(tmp[j1:j2, ...], (1, im_shape[1] / 512, im_shape[1] / 512), order=0)
            label = np.concatenate((label, tmp_label), axis=0)
            sample_caseID.append(i_case)
        tmp = tifffile.imread(case[:-8] + '*-im.tif')
        tmp = im_fix_width(tmp,  512)
        if im_shape[-1] == 1:
            tmp = np.expand_dims(tmp[::3, ...], axis=-1)
        else:
            tmp = np.moveaxis(np.reshape(tmp, (-1, 3,) + tmp.shape[1:]), 1, -1)
        for i in slice_list:
            j1, j2, j3, j4 = max(0, i - z_pad), i + z_pad + 1, max(0, z_pad - i), max(0, z_pad - i) + im_shape[0]
            tmp_im[0, j3:j4, ...] = zoom(tmp[j1:j2, ...], (1, im_shape[1] / 512, im_shape[1] / 512, 1))
            im = np.concatenate((im, tmp_im), axis=0)
    return im, label, sample_caseID


def load_train_data(folder_path, im_shape):
    # TODO: remove the hardcoded num_class
    num_class = 2
    im, label, sample_caseID = make_dataset(folder_path, im_shape)
    assert im.size > 0, "The data folder is empty: %s" % folder_path

    im = im.astype(np.float32) / 255
    label = (label == 3).astype(np.uint8)
    label = np.squeeze(label, axis=-1)
    if im_shape[0] == 1:
        im, label = np.squeeze(im, axis=1), np.squeeze(label, axis=1)
    label = np.reshape(np.squeeze(np.eye(num_class)[label.reshape(-1)]), label.shape + (num_class, ))
    train_data_id = np.nonzero(np.mod(sample_caseID, 2) == 1)[0]
    test_data_id = np.nonzero(np.mod(sample_caseID, 4) == 2)[0]
    valid_data_id = np.nonzero(np.mod(sample_caseID, 4) == 0)[0]
    return im, label, train_data_id, test_data_id, valid_data_id
