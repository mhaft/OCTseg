# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Convert the segmentation of layers to their boundaries"""

import numpy as np


def segmentation2boundary_slice(label):
    """Convert segmentation results to boundary results for a single slice.

    Segmentation are images in polar coordinate that foreground starts from zero-radius and can extend to any radius
    up to the size of image along the radius direction.  For each angle, the boundary is the fraction of the index
    of the last foreground pixel divided by the size of radius axis.  In case of no foreground pixel, returns -1.

    Args:
        label: A 2D numpy array, which first axis is radius and the second axis is angle.

    Returns:
        A vector of the boundary values at each angle.

    See Also:
        * :meth:`segmentation2boundary`
        * :meth:`boundary2segmentation`

    """
    nr, nc = label.shape
    out = np.zeros(nc)
    for i in range(nc):
        idx = np.flatnonzero(label[:, i])
        out[i] = np.max(idx) / nr if len(idx) else -1
    return out


def segmentation2boundary(label):
    """Convert segmentation results to boundary results.

    Segmentation are images in polar coordinate that foreground starts from zero-radius and can extend to any radius
    up to the size of image along the radius direction.  For each angle, the boundary is the fraction of the index
    of the last foreground pixel divided by the size of radius axis.  In case of no foreground pixel, returns -1.

    Args:
        label: inpput n dimensional numpy array.  The last axis is angle and the second axis from last is radius. The
        batch and depth axis can exist at the beginning of the array.

    Returns:
        The location of the last foreground at each degree as a fraction of the radius axis size.  The shape of of the
        output is similar to the `label`, except the size of radius axis is 1.

    See Also:
        * :meth:`segmentation2boundary_slice`
        * :meth:`boundary2segmentation`

    """
    label_shape = label.shape
    label = label.swapaxes(-1, -2).reshape((-1, label_shape[-2])).swapaxes(-1, -2)
    out = segmentation2boundary_slice(label)
    out = np.reshape(out, (label_shape[:-2] + (1, + label_shape[-1])))
    return out


def boundary2segmentation(boundary, w):
    """ converts a boundary tensor to a segmentation tensor.

    At each given angle, voxels below the boundary value will be foreground.  Boundary value is fraction of image
    width along the radius axis.

    Args:
        boundary: A ND numpy array. The last axis is angle.  The second axis from last is the boundary with size 1.
        Other axis can be batch or depth.
        w: size of radius axis in the output.

    Returns:
        A segmentation ND numpy array based on `boundary`

    See Also:
        * :meth:`segmentation2boundary_slice`
        * :meth:`segmentation2boundary`
    """
    out = np.tile(boundary.flatten() * w, (w, 1)).T
    out_range = np.tile(np.arange(w), (boundary.size, 1))
    return np.reshape(out >= out_range, boundary.shape[:-2] + (boundary.shape[-1], w)).swapaxes(-1, -2)


