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
    obj_list, last_row, last_case = {'lumen': [], 'iel': [], 'eel': [], 'gw': [], 'noniel': []}, [[''], ['']], ''
    lumen_label = ['lumen', 'fibro-fatty', 'fibrous', 'fc', 'fa', 'normal']
    gw_label = ['exclude', 'gw']
    ignore_label = ['', 'calcification', 'cap', 'calcium']
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        angle_label_balance = 0
        for i_row, row in enumerate(reader, 1):
            if len(row) > 4 and row[4] != '':
                row[4] = row[4].lower().strip()
                if row[4] in ['eel', 'iel', 'noniel']:
                    last_case = row[4]
                elif row[4] in gw_label:
                    last_case = 'gw'
                elif row[4] in lumen_label:
                    last_case = 'lumen'
                elif row[4] in ignore_label:
                    last_case = ''
                else:
                    raise Exception('Unknown label %s was found in the Line %d of file %s' % (row[4], i_row, file_path))
                if last_case != '':
                    obj_list[last_case].append([])
                if row[0] == 'Angle':
                    angle_label_balance += 3
            if row[0] in ['Line']:
                raise Exception('Unknown object %s was found in the Line %d of file %s' % (row[0], i_row, file_path))
            if i_row > 1 and last_row[0] != row[0] and (len(row) < 5 or row[4] == '') and \
                    (row[0] not in ['closed', 'open']):
                raise Exception('No label was found in the Line %d of file %s' % (i_row, file_path))
            last_row = row
            if row[0] == 'Angle':
                angle_label_balance -= 1
                if angle_label_balance < 0:
                    raise Exception('The angle object has a missing label in the Line %d of file %s' %
                                    (i_row, file_path))
            if last_case != '' and row[0] != 'closed' and row[0] != 'open':
                obj_list[last_case][-1].append([float(i) for i in row[1:4]])
            if last_case != '' and (row[0] == 'closed' or row[0] == 'open'):
                # TODO if there is real close this has to be changed to just for closed cases
                obj_list[last_case][-1].append(obj_list[last_case][-1][0])
    # n = len(obj_list['lumen'])
    # if len(obj_list['iel']) != n or len(obj_list['eel']) != n or len(obj_list['gw']) != n:
    #     raise Warning('There are same number of lumen (%d), IEL (%d), EEL (%d), and GW (%d) in this file (note there '
    #                   'is %d nonIEL): %s' % (len(obj_list['lumen']), len(obj_list['iel']), len(obj_list['eel']),
    #                                          len(obj_list['gw']), len(obj_list['noniel']), file_path))
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
    numPoint = len(obj_list)
    out = np.zeros(im_shape[-2:], dtype='int')
    # corner case for a boundary with less than three point.
    if numPoint < 2:
        return out

    # match 1-index to 0-index
    obj_list = np.array(obj_list) - 1

    # polar coordinate of each anchor point
    r, a = obj_list[:, 0],  obj_list[:, 1] * (2 * np.pi / im_shape[-1])

    # minimum number of points needed between consecutive pairs of anchor points
    min_arc_point = np.ceil(np.max((r[:-1] + r[1:]) / 2 * np.abs(a[:-1] - a[1:]))).astype(dtype='int') + 1

    # cartesian coordinates of the anchor points
    x, y = r * np.cos(a), r * np.sin(a)

    # rank order of the anchor points
    idx = np.arange(numPoint)

    # interpolation along the path
    fx = interp1d(idx, x, kind='quadratic')
    fy = interp1d(idx, y, kind='quadratic')
    x_ = fx(np.linspace(0, numPoint - 1, num=min_arc_point * numPoint))
    y_ = fy(np.linspace(0, numPoint - 1, num=min_arc_point * numPoint))
    r_ = np.round(np.sqrt(x_ ** 2 + y_ ** 2)).clip(0, im_shape[-2] - 1).astype('int')
    a_ = np.round((np.arctan2(- y_, - x_) / 2 / np.pi + 0.5) * im_shape[-1]).clip(0, im_shape[-1] - 1).astype('int')
    i = np.concatenate(([True], a_[:-2] != a_[1:-1], [(a_[0] != a[-1]) & (a_[-2] != a[-1])]))
    r_, a_ = r_[i], a_[i]

    # fill the boundary based on the all the interpolated points
    for i in range(len(a_)):
        out[:(r_[i] + 1), a_[i]] += 1

    return out > 0
    # return np.logical_and(out > 0, np.mod(out, 2) == np.mod(np.max(out), 2))


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

