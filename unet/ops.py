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
        * :meth:`KL.LeakyReLU`
        * :meth:`KL.Conv2D`
        * :meth:`KL.Conv3D`

    """
    ndims = len(x.get_shape()) - 2
    ConvND = getattr(KL, 'Conv%dD' % ndims)
    out_conv1 = ConvND(ChOut, kernel_size=3, padding='same', kernel_initializer='RandomNormal')(x)
    h_conv1 = KL.LeakyReLU()(out_conv1)
    out_conv2 = ConvND(ChOut, kernel_size=3, padding='same', kernel_initializer='RandomNormal')(h_conv1)
    h_conv2 = KL.LeakyReLU()(out_conv2)
    out_conv3 = ConvND(ChOut, kernel_size=3, padding='same', kernel_initializer='RandomNormal')(h_conv2)
    h_conv3 = KL.LeakyReLU()(out_conv3)
    return h_conv3


def MaxPoolingND(x):
    """Maxpooling in x and y direction for 2D and 3D inputs

    Args:
        x: input 4D or 5D tensor

    Returns:
        downscaled of `x` in x and y direction

    See Also:
        * :meth: `up_conv`
        * :meth: `KL.MaxPooling2D`
        * :meth: `KL.MaxPooling3D`

    """
    ndims = len(x.get_shape()) - 2
    MaxPoolingND = getattr(KL, 'MaxPooling%dD' % ndims)
    return MaxPoolingND(pool_size=(1,) * (ndims == 3) + (2, 2))(x)


def up_conv(x):
    """upscaling of input tensor in x and y direction using transpose convolution in 2D or 3D.

    Args:
        x: input 4D or 5D tensor

    Returns:
        unscaled of `x` in x and y direction

    See Also:
        * :meth: `MaxPoolingND`
        * :meth: `KL.Conv2DTranspose`
        * :meth: `KL.Conv3DTranspose`

    """
    x_shape = x.get_shape()
    ndims = len(x_shape) - 2
    ConvNDTranspose = getattr(KL, 'Conv%dDTranspose' % ndims)
    return ConvNDTranspose(x_shape[-1].value, (1,) * (ndims == 3) + (3, 3), strides=(1,) * (ndims == 3) + (2, 2),
                           padding='same',)(x)


def accuracy(labels, logits):
    """Measure accuracy metrics. The code calculate the prediction based on the input logits. Metrics are:

        * accuracy: The ratio of correctly labeled voxels to the total number of voxels.

        * Jaccard Index: ratio of number of foreground voxels in the intersection of `labels` and `logits` divided by
          total number of foreground voxels in the union of `labels` and `logits`

    .. math::
        accuracy &= \\frac{1}{N \\times M \\times L} \\sum_{i \\in [[N]],  j \\in [[M]],  k \\in [[L]]} (
            label_{i,j,k} == predict_{i,j,k}) \\\\
        Jaccard &= \\frac{
            \\sum_{i \\in [[N]],  j \\in [[M]],  k \\in [[L]]} (label_{i,j,k} \  \\&\\& \  predict_{i,j,k})
        }{
            \\sum_{i \\in [[N]],  j \\in [[M]],  k \\in [[L]]} (label_{i,j,k} \  \\| \  predict_{i,j,k})
        }

    Args:
        labels: 4D or 5D tensor of labels
        logits: 4D or 5D tensor of prediction logits.

    Returns:
        accuracy and Jaccard Index

    """
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1)), tf.float32), [1, 2])
    allButTN = tf.maximum(tf.argmax(logits, -1), tf.argmax(labels, -1))
    correct_prediction = tf.multiply(tf.argmax(logits, -1), tf.argmax(labels, -1))
    jaccard = tf.divide(tf.reduce_sum(tf.cast(correct_prediction, tf.float32)),
                        tf.reduce_sum(tf.cast(allButTN, tf.float32)))
    return accuracy, jaccard


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
        label = tf.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], outCh])
    else:
        image = tf.placeholder(tf.float32, shape=[None, im_shape[0], im_shape[1], im_shape[2], im_shape[3]])
        label = tf.placeholder(tf.float32, shape=[None, im_shape[0], im_shape[1], im_shape[2], outCh])
    return image, label
