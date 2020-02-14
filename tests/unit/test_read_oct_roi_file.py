# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Test read_oct_roi_file"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tifffile

from ...util.read_oct_roi_file import read_oct_roi_file, boundary_mask


def test_read_oct_roi_file():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures/test_read_oct_roi_file')
    im_shape = [5, 10, 10]
    out = read_oct_roi_file(file_path + 'ROI.txt', im_shape)
    tifffile.imwrite(file_path + '-out.tif', out)
    os.remove(file_path + '-out.tif')
    np.testing.assert_array_equal(out[1, 2, 1:4], [16, 17, 17])


def test_small_obj_list():
    # to test the corner case of small list
    _ = boundary_mask([[1, 1, 1]], (2, 10, 10, 2))
