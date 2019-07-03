# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized 
# copying of this file, via any medium is strictly prohibited Proprietary and 
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Convert an 2D or 3D image from polar or cylindrical coordinate to the 
    cartesian coordinate."""
    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def polar2cartesian(im, r0=0, full=True, deg=1, scale=1):
    if r0 >= 0:
        im = im[..., int(r0):, :]
    else:
        im = np.concatenate((im[..., -int(r0 + 1)::-1, :], im), axis=-2)
    if full:
        w = int(2 * im.shape[-2] * scale)
    else:
        w = int(2 * np.floor(im.shape[-2] / np.sqrt(2)) * scale)
    if len(im.shape) == 2:
        out = np.zeros((w, w), dtype=im.dtype)
    else:
        out = np.zeros((im.shape[0], w, w), dtype=im.dtype)
    c = w / 2 - 0.5
    x, y = np.unravel_index((np.arange(w * w)).astype(int), (w, w))
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2) / scale
    a = (np.arctan2(x - c, c - y) / np.pi + 1) * 180 * (im.shape[-1] / 360)
    valid_r = np.ceil(r) <= im.shape[-2]
    x, y, r, a = x[valid_r], y[valid_r], r[valid_r], a[valid_r]
    if deg == 0:
        r, a = (np.clip(np.round(r), 1, im.shape[-2] - 1)).astype(int), (np.mod(np.round(a), im.shape[-1])).astype(int)
        out[..., x, y] = im[..., r, a]
    else:
        dr, da = r - np.floor(r), a - np.floor(a)
        r, a = (np.clip(np.ceil(r), 1, im.shape[-2] - 1)).astype(int), np.mod(np.ceil(a).astype(int), im.shape[-1])
        m1, m2, m3, m4 = dr * da, dr * (1 - da), (1 - dr) * da, (1 - dr) * (1 - da)
        out[..., x, y] = m4 * im[..., r - 1, a - 1] + m3 * im[..., r - 1, a] + m2 * im[..., r, a - 1] + \
                          m1 * im[..., r, a]
    return out


def polar2cartesian_large_3d_file(im, r0=0, full=True, deg=1, scale=1):
    out = polar2cartesian(im[0:100, ...], r0=r0, full=full, deg=deg, scale=scale)
    for i in range(1, int(np.ceil((im.shape[0] - 1)/100)) + 1):
        sub_out = polar2cartesian(im[(i * 100):((i + 1) * 100), ...], r0=r0, full=full, deg=deg, scale=scale)
        out = np.concatenate((out, sub_out), axis=0)
    return out
