# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Convert an 2D or 3D image from polar or cylindrical coordinate to the
    cartesian coordinate."""

from __future__ import absolute_import, division, print_function

import os
import glob

import h5py
import tifffile
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom

from .process_oct_folder import process_oct_folder


def im_fix_width(im, w):
    """pad or crop the 3D image to have width and length equal to the input width in Cartesian coordinate system.

    Args:
        im: input image
        w: output width size

    Returns:
        image with the `w` width and length

    """
    w0 = im.shape[1]
    if w <= w0:
        i_start = (w0 - w) // 2
        im = im[:, i_start:(i_start + w), i_start:(i_start + w)]
    else:
        pad1 = (w - w0) // 2
        pad2 = w - w0 - pad1
        im = np.pad(im, ((0, 0), (pad1, pad2), (pad1, pad2)), 'constant', constant_values=0)
    return im


def make_dataset(folder_path, im_shape, coord_sys, carts_w=512):
    """Produce dataset based oon the results of :meth:`util.process_oct_folder`

    Args:
        folder_path: the path to the folder that contains the images
        im_shape: shape of the images in the dataset in (depth,width,length,channel) format
        coord_sys: coordinate system (`polar` or `carts`)
        carts_w: width of the image in case `coord_sys == carts`

    Returns:
        image and label as the 4D or 5D tensors.  sample_caseID contains the case ID for each row of image and label

    See Also:
        * :meth:`util.process_oct_folder`

    """
    # asserts
    assert len(im_shape) == 4, 'im_shape should have 4 element. got %d' % len(im_shape)
    assert coord_sys.lower() in ['polar', 'carts'], 'Coordinate system should be polar or carts. got %s' % coord_sys

    # set the placeholders
    tmp_im = np.zeros((1,) + tuple(im_shape), dtype='uint8')
    tmp_label = np.zeros((1,) + tuple(im_shape[:-1]) + (1,), dtype='uint8')
    im = np.zeros((0,) + tuple(im_shape), dtype='uint8')
    label = np.zeros((0,) + tuple(im_shape[:-1]) + (1,), dtype='uint8')
    # grab the right segmentation based on the coordinate system
    if not glob.glob(folder_path + '*-SegC.tif'):
        process_oct_folder(folder_path)
    if coord_sys.lower() == 'carts':
        cases = glob.glob(folder_path + '*-SegC.tif')
    elif coord_sys.lower() == 'polar':
        cases = glob.glob(folder_path + '*-SegP.tif')
    z_pad = (im_shape[0] - 1) // 2
    sample_caseID = []
    sample_sliceID = []
    i_patient, last_patient = -1, ""
    # iterate over the cases
    for case in tqdm(cases):
        # read segmentation
        tmp = tifffile.imread(case)
        if coord_sys == 'carts':
            tmp = im_fix_width(tmp, carts_w)
        # find the annotated slices
        slice_list = np.nonzero(np.any(np.any(tmp > 1, axis=-1), axis=-1))[0]

        # patient id
        current_patient = case[len(folder_path):]
        current_patient = current_patient[1:] if current_patient[0] == '[' else current_patient
        current_patient = current_patient.split(']')[0].split('_')[0].lower()
        if current_patient != last_patient:
            i_patient += 1
            last_patient = current_patient
            print('patient %d: %s ' % (i_patient + 1, current_patient))

        # iterate over the annotated slices
        for i in slice_list:
            # crop the section
            j1, j2, j3, j4 = max(0, i - z_pad), i + z_pad + 1, max(0, z_pad - i), max(0, z_pad - i) + im_shape[0]
            tmp_label[0, j3:j4, :, :, 0] = zoom(tmp[j1:j2, ...], (1, im_shape[1] / tmp.shape[1],
                                                                  im_shape[2] / tmp.shape[2]), order=0)
            # add the sample to the placeholder
            label = np.concatenate((label, tmp_label), axis=0)

            # track the rows caseID and sliceID
            sample_caseID.append(i_patient)
            sample_sliceID.append(i)

        # read the image
        if coord_sys == 'carts':
            tmp = tifffile.imread(case[:-9] + '-im.tif')
            tmp = im_fix_width(tmp, carts_w)
        elif coord_sys == 'polar':
            tmp = tifffile.imread(case[:-9] + '.pstif')

        # single channel vs multi channel
        if im_shape[-1] == 1:
            tmp = np.expand_dims(tmp[::3, ...], axis=-1)
        else:
            tmp = np.moveaxis(np.reshape(tmp, (-1, 3,) + tmp.shape[1:]), 1, -1)

        # crop the section
        for i in slice_list:
            j1, j2, j3, j4 = max(0, i - z_pad), i + z_pad + 1, max(0, z_pad - i), max(0, z_pad - i) + im_shape[0]
            tmp_im[0, j3:j4, ...] = zoom(tmp[j1:j2, ...], (1, im_shape[1] / tmp.shape[1], im_shape[2] /
                                                           tmp.shape[2], 1), order=2, mode='reflect')
            # add the sample to the placeholder
            im = np.concatenate((im, tmp_im), axis=0)
    print('%d patients and %d pullbacks' % (i_patient + 1, len(cases)))
    return im, label, np.array(sample_caseID), np.array(sample_sliceID)


def load_train_data(folder_path, im_shape, coord_sys, saveOutput=False):
    """ loading the training data.

    Args:
        folder_path: the input folder path containing the data
        im_shape: shape of the images in the dataset in (depth,width,length,channel) format
        coord_sys: coordinate system (`polar` or `carts`)
        saveOutput: save the output of the function to a h5 file

    Returns:
        :

        * **im**: image tensor of dataset with first row is sample ID

        * **label**: label tensor similar to `im`

        * **train_data_id**: row IDs of training samples

        * **test_data_id**: row IDs of testing samples

        * **valid_data_id**: row IDs of validation samples

        * **sample_caseID**: caseID of each row


    See Also:
        :meth:`make_dataset`

    """
    im, label, sample_caseID, sample_sliceID = make_dataset(folder_path, im_shape, coord_sys)
    assert im.size > 0, "The data folder is empty: %s" % folder_path
    im = im.astype(np.float32) / 255
    label = np.unpackbits(label.astype(np.uint8), axis=-1)[..., ::-1]
    if im_shape[0] == 1:
        im, label = np.squeeze(im, axis=1), np.squeeze(label, axis=1)
    # # 50% - 25% - 25%
    # train_data_id = np.nonzero(np.mod(sample_caseID, 2) == 1)[0]
    # test_data_id = np.nonzero(np.mod(sample_caseID, 4) == 2)[0]
    # valid_data_id = np.nonzero(np.mod(sample_caseID, 4) == 0)[0]
    # 80% - 10% - 10%
    train_data_id = np.nonzero(np.mod(sample_caseID, 10) > 1)[0]
    test_data_id = np.nonzero(np.mod(sample_caseID, 10) == 1)[0]
    valid_data_id = np.nonzero(np.mod(sample_caseID, 10) == 0)[0]

    if saveOutput:
        data_file = os.path.join(folder_path, 'Dataset ' + coord_sys + ' Z%d-L%d-W%d-C%d.h5' % im_shape)
        with h5py.File(data_file, 'w') as f:
            f.create_dataset('im', data=im)
            f.create_dataset('label', data=label)
            f.create_dataset('train_data_id', data=train_data_id)
            f.create_dataset('test_data_id', data=test_data_id)
            f.create_dataset('valid_data_id', data=valid_data_id)
            f.create_dataset('sample_caseID', data=sample_caseID)
            f.create_dataset('sample_sliceID', data=sample_sliceID)

    return im, label, train_data_id, test_data_id, valid_data_id, sample_caseID, sample_sliceID
