# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized 
# copying of this file, via any medium is strictly prohibited Proprietary and 
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Test polar2cartesian."""

from __future__ import absolute_import, division, print_function

import numpy as np

from ...util.polar2cartesian import polar2cartesian, polar2cartesian_large_3d_file


class TestPolar2cartesian:
    
    def test_output_size(self):
        """test if the output image size matches the input image size"""
        # 2D
        im_polar = np.zeros((10, 10))
        im_cartesian = polar2cartesian(im_polar)
        np.testing.assert_array_equal(im_cartesian.shape, [20, 20])
        # 3D
        im_polar = np.zeros((3, 5, 10))
        im_cartesian = polar2cartesian(im_polar)
        np.testing.assert_array_equal(im_cartesian.shape, [3, 10, 10])
        # r0 arg
        im_polar = np.zeros((10, 10))
        im_cartesian = polar2cartesian(im_polar, r0=5)
        np.testing.assert_array_equal(im_cartesian.shape, [10, 10])
        # full arg
        im_polar = np.zeros((10, 10))
        im_cartesian = polar2cartesian(im_polar, full=False)
        np.testing.assert_array_equal(im_cartesian.shape, [14, 14])
        # negative radius
        im_polar = np.zeros((10, 10))
        im_cartesian = polar2cartesian(im_polar, r0=-2)
        np.testing.assert_array_equal(im_cartesian.shape, [24, 24])
        # scale
        im_polar = np.zeros((10, 10))
        im_cartesian = polar2cartesian(im_polar, scale=0.25)
        np.testing.assert_array_equal(im_cartesian.shape, [5, 5])

    def test_input_image_with_shape(self):
        # box
        im_polar = np.ones((10, 10))
        out = np.ones((14, 14))
        im_cartesian = polar2cartesian(im_polar, full=False, deg=0)
        np.testing.assert_allclose(im_cartesian, out)
        # circle
        c = 9.5
        i, j = np.unravel_index((np.arange(20 * 20)).astype(int), (20, 20))
        out = (((i - c) ** 2 + (j - c) ** 2) <= 100).reshape((20, 20))
        im_cartesian = polar2cartesian(im_polar)
        np.testing.assert_allclose(im_cartesian, out)

    def test_polar2cartesian_large_3d_file(self):
        im_polar = np.ones((101, 3, 3))
        im_cartesian = polar2cartesian_large_3d_file(im_polar)
        np.testing.assert_array_equal(im_cartesian.shape, [101, 6, 6])
