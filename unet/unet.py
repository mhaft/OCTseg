# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Build unet model"""

import numpy as np
import tensorflow as tf

from unet.ops import conv, conv_bn_relu, max_pool, up_conv, weight_variable


def unet_model(x, nFeature=32, outCh=2):
    """ Build U-Net model.

    args:
        x: input placeholder

    Returns:
        y: output tensor
    """
    # encoder layer 1
    h_conv1a, W_conv1a, b_conv1a = conv_bn_relu(x, x.shape[-1].value, nFeature)
    h_conv1b, W_conv1b, b_conv1b = conv_bn_relu(h_conv1a, nFeature, 2 * nFeature)
    h_pool1 = max_pool(h_conv1b)
    # encoder layer 2
    h_conv2a, W_conv2a, b_conv2a = conv_bn_relu(h_pool1, 2 * nFeature, 2 * nFeature)
    h_conv2b, W_conv2b, b_conv2b = conv_bn_relu(h_conv2a, 2 * nFeature, 4 * nFeature)
    h_pool2 = max_pool(h_conv2b)
    # encoder layer 3
    h_conv3a, W_conv3a, b_conv3a = conv_bn_relu(h_pool2, 4 * nFeature, 4 * nFeature)
    h_conv3b, W_conv3b, b_conv3b = conv_bn_relu(h_conv3a, 4 * nFeature, 8 * nFeature)
    h_pool3 = max_pool(h_conv3b)
    # latent layer
    h_conv4a, W_conv4a, b_conv4a = conv_bn_relu(h_pool3, 8 * nFeature, 8 * nFeature)
    h_conv4b, W_conv4b, b_conv4b = conv_bn_relu(h_conv4a, 8 * nFeature, 16 * nFeature)
    h_up_conv4 = up_conv(h_conv4b)
    # decoder layer 1
    h_conv5a, W_conv5a, b_conv5a = conv_bn_relu(tf.concat([h_conv3b, h_up_conv4], -1), 24 * nFeature, 8 * nFeature)
    h_conv5b, W_conv5b, b_conv5b = conv_bn_relu(h_conv5a, 8 * nFeature, 8 * nFeature)
    h_up_conv5 = up_conv(h_conv5b)
    # decoder layer 2
    h_conv6a, W_conv6a, b_conv6a = conv_bn_relu(tf.concat([h_conv2b, h_up_conv5], -1), 12 * nFeature, 4 * nFeature)
    h_conv6b, W_conv6b, b_conv6b = conv_bn_relu(h_conv6a, 4 * nFeature, 4 * nFeature)
    h_up_conv6 = up_conv(h_conv6b)
    # decoder layer 3
    h_conv7a, W_conv7a, b_conv7a = conv_bn_relu(tf.concat([h_conv1b, h_up_conv6], -1), 6 * nFeature, 2 * nFeature)
    h_conv7b, W_conv7b, b_conv7b = conv_bn_relu(h_conv7a, 2 * nFeature, 2 * nFeature)
    if len(x.get_shape()) == 4:  # 2D
        W_y_conv = weight_variable([1, 1, 2 * nFeature, outCh])
    else:  # 3D
        W_y_conv = weight_variable([1, 1, 1, 2 * nFeature, outCh])
    y = conv(h_conv7b, W_y_conv)
    return y
