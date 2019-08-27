# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""test segmentation2boundary"""

import numpy as np

from ...util.segmentation2boundary import segmentation2boundary_slice, segmentation2boundary, boundary2segmentation


def test_segmentation2boundary_slice():
    m = np.triu(np.ones((4, 4)))
    m[:, -1] = 0
    out = segmentation2boundary_slice(m)
    np.testing.assert_array_equal(out, [0, .25, .5, -1])
    m = np.ones((4, 4))
    out = segmentation2boundary_slice(m)
    np.testing.assert_array_equal(out, [.75, .75, .75, .75])


def test_segmentation2boundary():
    m = np.concatenate((np.triu(np.ones((4, 4)))[np.newaxis, ...], np.ones((4, 4))[np.newaxis, ...]), axis=0)
    out = segmentation2boundary(m)
    np.testing.assert_array_equal(out.shape, [2, 1, 4])
    np.testing.assert_array_equal(out.flatten(), [0, .25, .5, .75, .75, .75, .75, .75])


def test_boundary2segmentation():
    m = np.triu(np.ones((4, 4)))
    m[:, -1] = 0
    m = np.concatenate((m[np.newaxis, ...], np.ones((4, 4))[np.newaxis, ...]), axis=0)
    out = segmentation2boundary(m)
    m_back = boundary2segmentation(out, m.shape[-2])
    np.testing.assert_array_equal(m, m_back)
