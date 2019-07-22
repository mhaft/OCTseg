# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""process OCT folder to generate the segmentation labels of cases. Each case all three -.PSTIF, -.INI, and -ROI.txt
    files"""

from __future__ import absolute_import, division, print_function

import glob
import csv
import tifffile
import numpy as np

from .polar2cartesian import polar2cartesian_large_3d_file
from .read_oct_roi_file import read_oct_roi_file


def process_oct_folder(folder_path, scale=0.25):
    cases = glob.glob(folder_path + '*.pstif')
    for case in cases:
        print(case[len(folder_path):])
        with open(case[:-6] + 'ROI.ini', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            r0 = 1
            for row in reader:
                if row[0] == 'zeroOffset':
                    # str to float to int
                    r0 = float(row[1]) - 1
        im = tifffile.imread(case)
        im_shape0 = im.shape
        im = polar2cartesian_large_3d_file(im, r0=r0, full=True, deg=1, scale=scale)
        tifffile.imwrite(case[:-6] + '-im.tif', im)

        seg = read_oct_roi_file(case[:-6] + 'ROI.txt', (int(im_shape0[0] / 3),) + im_shape0[1:])
        seg = polar2cartesian_large_3d_file(seg, r0=r0, full=True, deg=0, scale=scale)
        tifffile.imwrite(case[:-6] + '-Seg.tif', seg)
    return

