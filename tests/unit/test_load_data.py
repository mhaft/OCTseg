# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Test load_data"""


from __future__ import absolute_import, division, print_function

import os

from ...util.load_data import make_dataset, load_train_data


def test_make_dataset():
    # polar
    _ = make_dataset(os.path.dirname(os.path.realpath(__file__)) + '/fixtures/', (1, 10, 10, 1), coord_sys='polar')
    # cartesian with scale up
    _ = make_dataset(os.path.dirname(os.path.realpath(__file__)) + '/fixtures/', (1, 10, 10, 3), coord_sys='carts',
                     carts_w=15)
    # cartesian with scale down
    _ = make_dataset(os.path.dirname(os.path.realpath(__file__)) + '/fixtures/', (1, 10, 10, 3), coord_sys='carts',
                     carts_w=5)


def test_load_train_data():
    # load train data
    _ = load_train_data(os.path.dirname(os.path.realpath(__file__)) + '/fixtures/', (1, 10, 10, 1), coord_sys='polar')
