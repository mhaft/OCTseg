# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related operations"""

import keras.layers as KL
import tensorflow as tf


def conv_layer(x, ChOut):
    """Multi-layer convolution operators consists of three convolutions (2D or 3D based on the input shape) followed by
    LeakyReLY.

    Args:
        x: input 4D or 5D tensor to the layers
        ChOut: number of features of outputs of all convolutions

    Returns:
        output of the final layer with same size as `x`

    See Also:
        * :meth:`keras.layers.LeakyReLU`
        * :meth:`keras.layers.Conv2D`
        * :meth:`keras.layers.Conv3D`

    """
    ndims = len(x.get_shape()) - 2
    ConvND = getattr(KL, 'Conv%dD' % ndims)
    out_conv1 = ConvND(ChOut, kernel_size=3, padding='same', kernel_initializer='RandomNormal')(x)
    h_conv1 = KL.Activation('relu')(out_conv1)
    out_conv2 = ConvND(ChOut, kernel_size=3, padding='same', kernel_initializer='RandomNormal')(h_conv1)
    h_conv2 = KL.Activation('relu')(out_conv2)
    return h_conv2


def MaxPoolingND(x):
    """Maxpooling in x and y direction for 2D and 3D inputs

    Args:
        x: input 4D or 5D tensor

    Returns:
        downscaled of `x` in x and y direction

    See Also:
        * :meth:`up_conv`
        * :meth:`keras.layers.MaxPooling2D`
        * :meth:`keras.layers.MaxPooling3D`

    """
    ndims = len(x.get_shape()) - 2
    MaxPoolingND = getattr(KL, 'MaxPooling%dD' % ndims)
    return MaxPoolingND(pool_size=(1,) * (ndims == 3) + (2, 1))(x)


def placeholder_inputs(im_shape, outCh):
    """Generate placeholder variables to represent the input tensors.

    Args:
        im_shape: shape of the input tensor
        outCh: number of channels in the output

    Returns:
        image and label placeholders

    """
    if im_shape[0] == 1:
        image = tf.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], im_shape[3]])
        label = tf.placeholder(tf.float32, shape=[None, outCh, im_shape[2]])
    else:
        image = tf.placeholder(tf.float32, shape=[None, im_shape[0], im_shape[1], im_shape[2], im_shape[3]])
        label = tf.placeholder(tf.float32, shape=[None, im_shape[0], outCh, im_shape[2]])
    return image, label


def dense_layer(x, nFeature):
    """ A Dense layer in Keras.

    A dense layer in Keras with input `x` and outpur that has `nFeature` features.

    Args:
        x: input matrix
        nFeature: number of output features

    Returns:
        keras layer
    """
    return KL.Dense(nFeature, kernel_initializer='RandomNormal', activation='relu')(x)
