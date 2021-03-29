# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Post-processing the output labels"""

import cv2
import numpy as np


def clean_label(label, isOneObj=False, isColumn=False, isBoundary=False, filterSize=(5, 15)):
    c_label = label.astype('uint8')
    c_label = cv2.GaussianBlur(c_label.astype('uint8'), filterSize, 0)
    if isOneObj:
        cc = cv2.connectedComponentsWithStats(c_label, connectivity=8)
        c_label = (cc[1] == (1 + np.argmax(cc[2][1:, -1]))).astype('uint8')
    flood_label = np.logical_not(c_label).astype('uint8')
    i, j = np.nonzero(flood_label)
    if i.size:
        flood_mask = np.zeros((c_label.shape[-2] + 2, c_label.shape[-1] + 2), 'uint8')
        cv2.floodFill(flood_label, flood_mask, (i[-1], j[-1]), 0)
        c_label = np.logical_or(flood_label, c_label)
    if isColumn:
        c_label = np.tile(np.any(c_label, axis=-2, keepdims=True), (c_label.shape[-2], 1))
    if isBoundary:
        y = [np.max(np.nonzero(c_label[:, j])) if np.any(c_label[:, j]) else -1 for j in range(c_label.shape[-1])]
        c_label = (np.tile(y, (c_label.shape[-2], 1)) >=
                   np.tile(np.arange(c_label.shape[-2])[:, np.newaxis], (1, c_label.shape[-1])))
    return c_label


def segmentation2boundary_6x3(segmentation):
    lumen = segmentation == 1
    IEL = np.logical_or(segmentation == 2, lumen) * np.any(segmentation == 2, axis=-2, keepdims=True)
    EEL = np.logical_or(segmentation == 3, IEL) * np.any(segmentation == 3, axis=-2, keepdims=True)
    return lumen, IEL, EEL


def clean_6label(label, filterSize=(5, 15)):
    def im_open(im, itr=2):
        for _ in range(itr):
            im = cv2.dilate(cv2.erode(im.astype('uint8'), np.ones((1, filterSize[1] - 2))), np.ones((1, filterSize[1])))
        return im

    if label.ndim == 3:
        return np.concatenate(tuple(np.expand_dims(clean_6label(label[i, ...], filterSize=filterSize), 0)
                                    for i in range(label.shape[0])), axis=0)

    c0 = np.logical_not(clean_label(label > 0, isOneObj=True, isColumn=False, isBoundary=True,
                                    filterSize=(2 * filterSize[0] + 1, filterSize[1])))
    label[c0] = 0
    c1 = clean_label(label == 1, isOneObj=True, isColumn=False, isBoundary=True, filterSize=filterSize)
    c1 = np.logical_and(c1, np.logical_not(c0))
    c4 = clean_label(label == 4, isOneObj=False, isColumn=True, isBoundary=False, filterSize=filterSize)
    c5 = clean_label(np.logical_or(label == 4, label == 5), isOneObj=False, isColumn=True,
                     isBoundary=False, filterSize=filterSize)
    c_2_3_mask = np.logical_not(np.logical_and(
        clean_label(label == 2, isOneObj=False, isColumn=True, isBoundary=False, filterSize=filterSize),
        clean_label(label == 3, isOneObj=False, isColumn=True, isBoundary=False, filterSize=filterSize)))
    c4, c5, c_2_3_mask = im_open(c4, itr=2), im_open(c5, itr=2), im_open(c_2_3_mask, itr=4)
    c4 = np.logical_and(c4, c_2_3_mask)
    c5 = np.logical_and(c5, c_2_3_mask) + c4
    c2 = clean_label(label == 2, isOneObj=False, isColumn=False, isBoundary=True, filterSize=filterSize)
    c_3_0 = np.logical_not(clean_label(np.logical_not(np.logical_or(label == 3, label == 0)),
                                       isOneObj=True, isColumn=False, isBoundary=True, filterSize=filterSize))
    c2 = np.logical_and(c2, np.logical_not(c0 + c1 + c5 + c_3_0))
    c4 = np.logical_and(c4, np.logical_not(c1 + c0))
    c5 = np.logical_and(c5, np.logical_not(c0 + c1 + c4)) + \
         np.logical_and(np.logical_not(np.any(c2, axis=-2, keepdims=True)), np.logical_not(c0 + c1 + c4))
    c3 = np.logical_not(c0 + c1 + c2 + c4 + c5)
    return (c1 + 2 * c2 + 3 * c3 + 4 * c4 + 5 * c5).astype(label.dtype)
