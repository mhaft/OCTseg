# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related operations"""

import keras.layers as KL
import tensorflow as tf


def conv_layer(x, ChOut):
    ndims = len(x.get_shape()) - 2
    ConvND = getattr(KL, 'Conv%dD' % ndims)

    out_conv1 = ConvND(ChOut, kernel_size=3, padding='same', kernel_initializer='RandomNormal')(x)
    h_conv1 = KL.LeakyReLU()(out_conv1)
    out_conv2 = ConvND(ChOut, kernel_size=3, padding='same', kernel_initializer='RandomNormal')(h_conv1)
    h_conv2 = KL.LeakyReLU()(out_conv2)
    return h_conv2


def MaxPoolingND(x):
    ndims = len(x.get_shape()) - 2
    MaxPoolingND = getattr(KL, 'MaxPooling%dD' % ndims)
    return MaxPoolingND(pool_size=(1,) * (ndims == 3) + (2, 2))(x)


def up_conv(x):
    x_shape = x.get_shape()
    ndims = len(x_shape) - 2
    ConvNDTranspose = getattr(KL, 'Conv%dDTranspose' % ndims)
    return ConvNDTranspose(x_shape[-1].value, (1,) * (ndims == 3) + (3, 3), strides=(1,) * (ndims == 3) + (2, 2),
                           padding='same',)(x)


def accuracy(labels, logits):
    """measure accuracy metrics"""
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1)), tf.float32), [1, 2])
    allButTN = tf.maximum(tf.argmax(logits, -1), tf.argmax(labels, -1))
    correct_prediction = tf.multiply(tf.argmax(logits, -1), tf.argmax(labels, -1))
    jaccard = tf.divide(tf.reduce_sum(tf.cast(correct_prediction, tf.float32)),
                        tf.reduce_sum(tf.cast(allButTN, tf.float32)))
    return accuracy, jaccard


def placeholder_inputs(im_shape, outCh):
    """Generate placeholder variables to represent the input tensors."""
    if im_shape[0] == 1:
        image = tf.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], im_shape[3]])
        label = tf.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], outCh])
    else:
        image = tf.placeholder(tf.float32, shape=[None, im_shape[0], im_shape[1], im_shape[2], im_shape[3]])
        label = tf.placeholder(tf.float32, shape=[None, im_shape[0], im_shape[1], im_shape[2], outCh])
    return image, label
