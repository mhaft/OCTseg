# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Build U-Net model"""

import keras.layers as KL
from keras.models import Model
from unet.ops import conv_layer, up_conv, MaxPoolingND


def unet_model(im_shape, nFeature=32, outCh=2, nLayer=3, pool_scale=2):
    """ Build U-Net model.

        Args:
            x: input placeholder
            outCh: number of output channels

        Returns:
            keras model

    """
    if im_shape[0] == 1:
        im_shape = im_shape[1:]
    x = KL.Input(shape=im_shape)
    out = [x]
    for iLayer in range(nLayer):
        h = conv_layer(out[-1], (2 ** (iLayer // 2)) * nFeature)
        out.append(MaxPoolingND(h, s=pool_scale))

    out.append(conv_layer(out[-1], (2 ** (nLayer // 2)) * nFeature))
    out.append(conv_layer(out[-1], (2 ** (nLayer // 2)) * nFeature))

    for iLayer in range(nLayer - 1, -1, -1):
        u = up_conv(out[-1], s=pool_scale)
        if iLayer == 0:
            u = KL.Cropping2D(cropping=((0, (im_shape[-3] + pool_scale - 1) // pool_scale * pool_scale - im_shape[-3]),
                                        (0, (im_shape[-2] + pool_scale - 1) // pool_scale * pool_scale - im_shape[-2])),
                              data_format=None)(u)

        c = KL.Concatenate()([out[iLayer], u])
        out.append(conv_layer(c, (2 ** (iLayer // 2)) * nFeature))

    out.append(conv_layer(out[-1], outCh))

    return Model(inputs=x, outputs=out[-1])

