# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Build boundary model"""

import keras.layers as KL
from keras.models import Model
import keras.backend as K

from boundary.ops import conv_layer, dense_layer


def boundary_model(im_shape, nFeature=16, outCh=2, nLayer=3):
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
        h = conv_layer(out[-1], nFeature)
        if iLayer:
            out.append(KL.add([h, out[-1]]))
        else:
            out.append(h)

    out.append(KL.Lambda(lambda i: K.permute_dimensions(i, (0, 2, 1, 3)))(out[-1]))
    out.append(KL.Lambda(lambda i: K.reshape(i, (-1, im_shape[1], im_shape[0] * nFeature)))(out[-1]))
    for iLayer in range(nLayer):
        out.append(dense_layer(out[-1], im_shape[0] * nFeature))
    out.append(dense_layer(out[-1], outCh))
    out.append(KL.Lambda(lambda i: K.permute_dimensions(i, (0, 2, 1)))(out[-1]))

    return Model(inputs=x, outputs=out[-1])

