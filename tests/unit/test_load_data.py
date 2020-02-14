# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Test load_data"""


from __future__ import absolute_import, division, print_function

import os

from ...util.load_data import make_dataset, load_train_data
from ...util.process_oct_folder import process_oct_folder


def test_make_dataset():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures', '')
    process_oct_folder(file_path)

    # polar
    _ = make_dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures', ''), (1, 10, 10, 1),
                     coord_sys='polar')
    # cartesian with scale up
    _ = make_dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures', ''), (1, 10, 10, 3),
                                  coord_sys='carts', carts_w=15)
    # cartesian with scale down
    _ = make_dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures', ''), (1, 10, 10, 3),
                                  coord_sys='carts', carts_w=5)
    os.remove(os.path.join(file_path, 'test_read_oct_roi_file-SegC.tif'))
    os.remove(os.path.join(file_path, 'test_read_oct_roi_file-SegP.tif'))


def test_load_train_data():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures', '')
    process_oct_folder(file_path)
    # load train data
    _ = load_train_data(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fixtures', ''), (1, 10, 10, 1),
                                     coord_sys='polar')
    os.remove(os.path.join(file_path, 'test_read_oct_roi_file-SegC.tif'))
    os.remove(os.path.join(file_path, 'test_read_oct_roi_file-SegP.tif'))
