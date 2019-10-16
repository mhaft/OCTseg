# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Read ROI file generated based on the and generate segmentation results."""

from __future__ import absolute_import, division, print_function

import csv
import numpy as np

from scipy.interpolate import interp1d


def roi_file_parser(file_path):
    """Parse roi file and output the lists of objects.

    The object list is a dictionary that has a list for each keys.  Each class of object has a key.  Keys are:

    * **lumen**: vessel lumen boundary

    * **iel**: IEL boundary

    * **gw**: guide wire arc defined by three points.

    * **noniel**: None IEL arc, which is the places that IEL is not visible, defined by three points

    Each key's value is a nested list, which first index represents a complete boundary or an arc whitin a plane.
    The second index represents a point within the boundary or arc.  The third index represnets the x, y, z
    coordinates of the point.

    Notes:
        In `*ROI.txt` files:
            * Each file has a header line as *ROIformat*.
            * The first record of a boundary or an arc section has a final field that contain the classification label.
            * Each boundary record section finishes with the keyword *closed*.
            * Boundary records start with keyword *Snake*.
            * Arc records start with keyword *Angle*.
            * Lumen class can have one of the following classification labels:
                #. *lumen*
                #. *fibro-fatty*
                #. *fibrous*
                #. *fc*
                #. *fibrous*
                #. *fa*
                #. *normal*
            * There are some classifications that are ignore in this function such as *calcification*

    Args:
        file_path: the path to `*ROI.txt` file

    Returns:
        The object_list dictionary.

    See Also:
        * :meth:`read_oct_roi_file`

    """
    obj_list, last_row, last_case = {'lumen': [], 'iel': [], 'eel': [], 'gw': [], 'noniel': []}, [''], ''
    lumen_label = ['lumen', 'fibro-fatty', 'fibrous', 'fc', 'fa', 'normal']
    gw_label = ['exclude', 'gw']
    ignore_label = ['', 'calcification', 'cap', 'calcium']
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > 4 and row[4] != '':
                row[4] = row[4].lower()
                if row[4] in ['eel', 'iel', 'noniel']:
                    last_case = row[4]
                elif row[4] in gw_label:
                    last_case = 'gw'
                elif row[4] in lumen_label:
                    last_case = 'lumen'
                elif row[4] in ignore_label:
                    last_case = ''
                else:
                    raise Exception('unknown label %s was found in the file %s' % (row[4], file_path))
                if last_case != '':
                    obj_list[last_case].append([])
            if last_case != '' and row[0] != 'closed' and row[0] != 'open':
                obj_list[last_case][-1].append([float(i) for i in row[1:4]])
    return obj_list


def boundary_mask(obj_list, im_shape):
    """generate lumen or IEL mask based on the point list.

    Based on the periodic nature of polar coordinate system, the boundary within [0, 2 pi] is copied to [- 2 pi, 0] and
    [2 pi, 4 pi]. The interpolation happened along the path for x and y independently. Number of points interpolated
    between each two consecutive points is based on the largest arc length between all pairs of consecutive points
    measured in the cartesian coordinate system.

    Args:
        obj_list: list of a single boundary in a single plane
        im_shape: the original image shape

    Returns:
        A mask image based on the `obj_list` and size of `im_obj_list`.

    See Also:
        * :meth:`read_oct_roi_file`

    """
    out = np.zeros(im_shape[-2:], dtype='bool')
    # corner case for a boundary with less than three point.
    if len(obj_list) < 2:
        return out

    # match 1-index to 0-index
    obj_list = np.array(obj_list) - 1

    # the boundary within [0, 2 pi] is copied to [- 2pi, 0] and [2 pi, 4 pi]
    obj_list = np.concatenate((obj_list - [0, im_shape[-1], 0], obj_list, obj_list +
                               [0, im_shape[-1], 0]), axis=0)

    # polar coordinate of each anchor point
    r, a = obj_list[:, 0],  obj_list[:, 1] * (2 * np.pi / im_shape[-1])

    # minimum distance needed between interpolated points between consecutive pairs of anchor points
    min_arc_dist = 1.0 / np.max((r[:-1] + r[1:]) / 2 * np.abs(a[:-1] - a[1:]) + 1)

    # cartesian coordinates of the anchor points
    x, y = r * np.cos(a), r * np.sin(a)

    # rank order of the anchor points with 0 for the first recorder anchor point and -1 is the last anchor point but
    # copied to the [- 2 pi, 0] span.
    idx = np.arange(obj_list.shape[0]) - obj_list.shape[0] // 3

    # interpolation along the path
    fx = interp1d(idx, x, kind='quadratic')
    fy = interp1d(idx, y, kind='quadratic')
    x_ = fx(np.arange(0, obj_list.shape[0] // 3 + 1, min_arc_dist))
    y_ = fy(np.arange(0, obj_list.shape[0] // 3 + 1, min_arc_dist))
    r_ = np.round(np.sqrt(x_ ** 2 + y_ ** 2)).clip(0, im_shape[-2] - 1).astype('int')
    a_ = np.round((np.arctan2(- y_, - x_) / 2 / np.pi + 0.5) * im_shape[-1]).clip(0, im_shape[-1] - 1).astype('int')

    # fill the boundary based on the all the interpolated points
    for i in range(len(a_)):
        out[:(r_[i] + 1), a_[i]] = True

    return out


def read_oct_roi_file(file_path, im_shape):
    """ generate a label tensor based on a `*ROI.txt` file.  The label tensor is a 8-bit integer, which each bit
    encode one the classes:

        * bit 1 (2**0) encode `gw` (Guide Wire, where guide wire has shadow)
        * bit 2 (2**1) encode `noniel` (NonIEL, where IEL is not visible)
        * bit 2  (2**2) encode `lumen` area
        * bit 3  (2**3) encode `iel` area (the *Tunica Intima* layer), which is the layer between `lumen` and `iel`
            boundaries.
        * bit 4 (2**4) encode `eel` area (the *Tunica Media* layer), which is the layer between `iel` and `eel`
            boundaries
        * other bits are not utilized and can be used for other classes in future.

    Args:
        file_path: file path to `*ROI.txt` file for a case
        im_shape: the output image shape.

    Returns:
        uint8: the output label image

    See Also:
        * :meth:`boundary_mask`
        * :meth:`roi_file_parser`

    """
    obj_list = roi_file_parser(file_path)
    out = np.zeros(tuple(im_shape) + (8,), dtype=np.uint8)
    for eel in obj_list['eel']:
        z = int(eel[0][2]) - 1
        if z < out.shape[0]:
            slice2D = out[z, ..., 4]
            slice2D[boundary_mask(eel, im_shape)] = 1
            out[z, ..., 4] = slice2D
    for iel in obj_list['iel']:
        z = int(iel[0][2]) - 1
        if z < out.shape[0]:
            slice2D = out[z, ..., 3]
            slice2D[boundary_mask(iel, im_shape)] = 1
            out[z, ..., 3] = slice2D
    for lumen in obj_list['lumen']:
        z = int(lumen[0][2]) - 1
        if z < out.shape[0]:
            slice2D = out[z, ..., 2]
            slice2D[boundary_mask(lumen, im_shape)] += 1
            out[z, ..., 2] = slice2D
    for i, gw in enumerate(obj_list['gw'] + obj_list['noniel'], 1):
        z = int(gw[1][2]) - 1
        # val is 1 and 2 for GW and NonIEL, respectively
        val = 1 if (i > len(obj_list['gw'])) else 0
        gw = np.array([gw[1][1], gw[2][1]]).astype('int')
        if z < out.shape[0]:
            if gw[0] <= gw[1]:
                out[z, :, gw[0]:(gw[1] + 1), val] = 1
            else:
                out[z, :, gw[0]:, val] = 1
                out[z, :, :(gw[1] + 1), val] = 1

    # makes the layers exclusive
    out[..., 4] *= 1 - out[..., 3]
    out[..., 3] *= 1 - out[..., 2]
    out = np.packbits(out[..., ::-1], axis=-1)[..., 0]
    return out

