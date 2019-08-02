# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Build unet model"""

import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
from unet.ops import conv_layer, up_conv, MaxPoolingND


def unet_model(im_shape, nFeature=32, outCh=2):
    """ Build U-Net model.

        Arguments:
            x: input placeholder
            outCh: number of output channels

        Returns:
            keras model

    """
    if im_shape[0] == 1:
        im_shape = im_shape[1:]
    x = KL.Input(shape=im_shape)
    h1 = conv_layer(x, nFeature)
    p1 = MaxPoolingND(h1)
    h2 = conv_layer(p1, 2 * nFeature)
    p2 = MaxPoolingND(h2)
    h3 = conv_layer(p2, 4 * nFeature)
    p3 = MaxPoolingND(h3)
    h4 = conv_layer(p3, 8 * nFeature)
    u4 = up_conv(h4)
    c5 = KL.Concatenate()([h3, u4])
    h5 = conv_layer(c5, 6 * nFeature)
    u5 = up_conv(h5)
    c6 = KL.Concatenate()([h2, u5])
    h6 = conv_layer(c6, 3 * nFeature)
    u6 = up_conv(h6)
    c7 = KL.Concatenate()([h1, u6])
    h7 = conv_layer(c7, nFeature)
    out = conv_layer(h7, outCh)
    return Model(inputs=x, outputs=out)

