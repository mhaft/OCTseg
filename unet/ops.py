# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related operations"""

import numpy as np
import keras.layers as KL
import tensorflow as tf
from scipy.ndimage import zoom


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


def img_rand_scale(im, scale, order):
    """scale one image batch"""
    out = np.zeros_like(im)
    new_shape = (len(im.shape) == 4) * (1,) + (scale, scale, 1)
    tmp = zoom(im, new_shape, order=order, mode='reflect')
    pad1, pad2 = (tmp.shape[-3] - im.shape[-3]) // 2, (tmp.shape[-2] - im.shape[-2]) // 2
    if scale > 1:
        i1, i2, j1, j2 = pad1, pad1 + im.shape[-3], pad2, pad2 + im.shape[-2]
        out = tmp[..., i1:i2, j1:j2, :]
    else:
        i1, i2, j1, j2 = - pad1, - pad1 + tmp.shape[-3], - pad2, - pad2 + tmp.shape[-2]
        out[..., i1:i2, j1:j2, :] = tmp
    return out


def img_aug(im, l, coord_sys):
    assert coord_sys in ['carts', 'polar'], 'the coord_sys should be carts or polar. got %d' % coord_sys
    if coord_sys == 'carts':
        return img_aug_carts(im, l)
    else:
        return img_aug_polar(im, l)


def img_aug_carts(im, l):
    """Data augmentation in Cartesian"""
    dim = len(im.shape) - 2
    p_lim = 0.9
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
        if np.random.rand() > p_lim:  # intensity scaling
            im_ = np.clip(im_ * (1 + 0.5 * (np.random.rand() - 0.5)), 0, 1)
        if np.random.rand() > p_lim:  # image scaling
            scale = 1 + 0.25 * (np.random.rand() - 0.5)
            im_ = img_rand_scale(im_, scale, 2)
            l_ = img_rand_scale(l_, scale, 0)
        im[i, ...], l[i, ...] = im_, l_
    return im, l


def img_aug_polar(im, l):
    """Data augmentation in Polar coordinate"""
    dim = len(im.shape) - 2
    p_lim = 0.9
    for i in range(im.shape[0]):
        im_, l_ = im[i, ...], l[i, ...]
        if np.random.rand() > p_lim:  # random rotation
            j = np.floor(np.random.rand() * im_.shape[-2]).astype('int64')
            im_ = np.concatenate((im_[..., j:, :], im_[..., :j, :]), axis=-2)
            l_ = np.concatenate((l_[..., j:, :], l_[..., :j, :]), axis=-2)
        if np.random.rand() > p_lim:  # intensity scaling
            im_ = np.clip(im_ * (1 + 0.5 * (np.random.rand() - 0.5)), 0, 1)
        if np.random.rand() > p_lim:  # image scaling
            scale = 1 + 0.25 * (np.random.rand() - 0.5)
            if scale > 1:
                tmp = np.zeros_like(im_)
                tmp = zoom(tmp, (dim == 3) * (1,) + (scale - 1, 1, 1))
                im_ = np.concatenate((im_, tmp), axis=-3)
                tmp = np.zeros_like(l_)
                tmp = zoom(tmp, (dim == 3) * (1,) + (scale - 1, 1, 1))
                l_ = np.concatenate((l_, tmp), axis=-3)
            else:
                j = np.ceil(im_.shape[-3] * scale).astype('int64')
                im_ = im_[..., :j, :, :]
                l_ = l_[..., :j, :, :]
            im_ = zoom(im_, (dim == 3) * (1,) + (im.shape[-3] / im_.shape[-3], 1, 1), order=2, mode='reflect')
            l_ = zoom(l_, (dim == 3) * (1,) + (l.shape[-3] / l_.shape[-3], 1, 1), order=0)
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


def load_batch(im, datasetID, nBatch, label=None, isAug=False, coord_sys='carts'):

    while True:
        j = np.random.randint(0, len(datasetID), nBatch)
        im_ = im[datasetID[j], ...]
        if label is not None:
            label_ = label[datasetID[j], ...]
        else:
            label_ = None
        if isAug:
            im_, label_ = img_aug(im_, label_, coord_sys)
        yield (im_, label_)
