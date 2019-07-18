# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related operations"""

import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv(x, W):
    if len(x.get_shape()) == 4:  # 2D
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    else:  # 3D
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool(x):
    dim = len(x.get_shape()) - 2
    if len(x.get_shape()) == 4:  # 2D
        return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    else:  # 3D
        return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                            strides=[1, 1, 2, 2, 1], padding='SAME')


def conv_bn_relu(x, ChIn, ChOut):
    if len(x.get_shape()) == 4:  # 2D
        W_conv = weight_variable([3, 3, ChIn, ChOut])
    else:
        W_conv = weight_variable([3, 3, 3, ChIn, ChOut])
    b_conv = bias_variable([ChOut])
    # h_conv = tf.nn.leaky_relu(tf.keras.layers.BatchNormalization()(conv2d(x, W_conv) + b_conv))
    h_conv = tf.nn.leaky_relu(conv(x, W_conv) + b_conv)
    return h_conv, W_conv, b_conv


def up_conv(x):
    x_shape = x.get_shape()
    # w = weight_variable([2, 2, x_shape[3].value, x_shape[3].value])
    # return tf.nn.conv2d_transpose(x, filter=w, output_shape=[2, 2 * x_shape[1].value,
    #                                             2 * x_shape[2].value, x_shape[3].value], strides=2)
    if len(x_shape) == 4:  # 2D
        return tf.keras.layers.Conv2DTranspose(x_shape[-1].value, 2, 2)(x)
    else:  # 3D
        return tf.keras.layers.Conv3DTranspose(x_shape[-1].value, (1, 2, 2), (1, 2, 2))(x)


def img_aug(im, l):
    """Data augmentation"""
    dim = len(im.shape) - 2
    p_lim = 0.05
    for i in range(im.shape[0]):
        im_, l_ = im[i, ...], l[i, ...]
        if np.random.rand() > p_lim:  # y=x mirror
            im_ = im_.swapaxes(-2, -3)
            l_ = l_.swapaxes(-2, -3)
        if np.random.rand() > p_lim:  # x mirror
            im_ = im_[..., ::-1, :]
            l_ = l_[..., ::-1, :]
        if np.random.rand() > p_lim:  # y mirror
            im_ = im_[..., ::-1, :, :]
            l_ = l_[..., ::-1, :, :]
        if np.random.rand() > p_lim and dim == 3:  # z mirror
            im_ = im_[::-1, :, :, :]
            l_ = l_[::-1, :, :, :]
        if np.random.rand() > p_lim:  # 1st 90 deg rotation
            im_ = np.rot90(im_, k=1, axes=(-2, -3))
            l_ = np.rot90(l_, k=1, axes=(-2, -3))
        if np.random.rand() > p_lim:  # 2nd 90 degree rotation
            im_ = np.rot90(im_, k=1, axes=(-2, -3))
            l_ = np.rot90(l_, k=1, axes=(-2, -3))
        if np.random.rand() > p_lim:  # 3rd 90 degree rotation
            im_ = np.rot90(im_, k=1, axes=(-2, -3))
            l_ = np.rot90(l_, k=1, axes=(-2, -3))
        # if np.random.rand() > p_lim:  # salt-and-pepper noise
        #     im_ = im_ + 0.01 * (np.random.rand() - 0.5)
        im[i, ...], l[i, ...] = im_, l_
    return im, l


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


def load_batch(im, datasetID, nBatch, label=None, iBatch=None, isAug=False):
    if iBatch is None:
        j = np.random.randint(0, len(datasetID), nBatch)
        im = im[datasetID[j], ...]
        if label is not None:
            label = label[datasetID[j], ...]
    else:
        j1, j2 = (iBatch * nBatch), ((iBatch + 1) * nBatch)
        im = im[datasetID[j1:j2], ...]
        if label is not None:
            label = label[datasetID[j1:j2], ...]
    if isAug:
        im, label = img_aug(im, label)
    return im, label
