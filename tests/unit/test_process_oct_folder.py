# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Test process_oct_folder.py"""

from __future__ import absolute_import, division, print_function

import os

from ...util.process_oct_folder import process_oct_folder


def test_process_oct_folder():
    process_oct_folder(os.path.dirname(os.path.realpath(__file__)) + '/fixtures/')
