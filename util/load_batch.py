# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Load a batch of data.

    Creates batches of data randomly in serial or multi-thread parallel fashion.

"""

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from scipy.ndimage import zoom
from keras.utils import Sequence


def img_rand_scale(im, scale, order):
    """Scale one image or label batch in Cartesian coordinate system.

    scale the image based on the input scale value and interpolation order followed by cropping or padding to
    maintain the original image shape.  For interpolation close to the boundaries,  the reflection mode is used.

    Args:
        im: 3D or 4D image or label tensor
        scale: scalar scale values for x and y direction
        order: interpolation order

    Returns:
        same size image with the scale image in the center of it.

    """
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


def img_aug(im, l, coord_sys, prob_lim=0.5):
    """Image augmentation manager.

    Based on the coordinate system (*Polar* vs.  *Cartesian*),  it selects the corresponding method.

    Args:
        im: input image 4D or 5D tensor
        l: input label 4D or 5D tensor
        coord_sys: coordinate system.  'polar' or 'carts' for Polar and Cartesian,  respectively.
        prob_lim: probability limit for applying each augmentation case.

    Returns:
        augmented im and l

    See Also:
        * :meth:`img_aug_carts`
        * :meth:`img_aug_polar`

    """
    assert coord_sys in ['carts', 'polar'], 'the coord_sys should be carts or polar. got %d' % coord_sys
    if coord_sys == 'carts':
        return img_aug_carts(im, l, prob_lim)
    else:
        return img_aug_polar(im, l, prob_lim)


def img_aug_carts(image, L, prob_lim=0.5):
    """Data augmentation in Cartesian coordinate system.

    Applies different image augmentation procedures:

        *  mirroring the image along 45 degree (y=x line)

        *  mirroring the image along the x axis

        *  mirroring the image along the y axis

        * mirroring the image along the z axis for 3D images

        * multiple 90 degree rotations

        * image intensity scaling by multplying the intensity values with close to one scale value

        * image scaling.  See :meth:`img_rand_scale`

     based on the input probability limit probabilistically applies different augmentation cases.

    Args:
        image: input image 4D or 5D tensor
        L: input label 4D or 5D tensor
        prob_lim: probability limit for applying each augmentation case.

    Returns:
        augmented image and L

    See Also:
        * :meth:`img_aug`

    """
    dim = len(image.shape) - 2
    for i in range(image.shape[0]):
        im_, l_ = image[i, ...], L[i, ...]
        if np.random.rand() > prob_lim:  # y=x mirror
            im_ = im_.swapaxes(-2, -3)
            l_ = l_.swapaxes(-2, -3)
        if np.random.rand() > prob_lim:  # x mirror
            im_ = im_[..., ::-1, :]
            l_ = l_[..., ::-1, :]
        if np.random.rand() > prob_lim:  # y mirror
            im_ = im_[..., ::-1, :, :]
            l_ = l_[..., ::-1, :, :]
        if np.random.rand() > prob_lim and dim == 3:  # z mirror
            im_ = im_[::-1, :, :, :]
            l_ = l_[::-1, :, :, :]
        if np.random.rand() > prob_lim:  # 1st 90 deg rotation
            im_ = np.rot90(im_, k=1, axes=(-2, -3))
            l_ = np.rot90(l_, k=1, axes=(-2, -3))
        if np.random.rand() > prob_lim:  # 2nd 90 degree rotation
            im_ = np.rot90(im_, k=1, axes=(-2, -3))
            l_ = np.rot90(l_, k=1, axes=(-2, -3))
        if np.random.rand() > prob_lim:  # 3rd 90 degree rotation
            im_ = np.rot90(im_, k=1, axes=(-2, -3))
            l_ = np.rot90(l_, k=1, axes=(-2, -3))
        if np.random.rand() > prob_lim:  # intensity scaling
            im_ = np.clip(0.1 * (np.random.rand() - 0.5) + im_ * (1 + 0.20 * (np.random.rand() - 0.5)), 0, 1)
        if np.random.rand() > prob_lim:  # image scaling
            scale = 1 + 0.25 * (np.random.rand() - 0.5)
            im_ = img_rand_scale(im_, scale, 2)
            l_ = img_rand_scale(l_, scale, 0)
        image[i, ...], L[i, ...] = im_, l_
    return image, L


def img_aug_polar(image, label, prob_lim=0.5):
    """Data augmentation in Polar coordinate.

    Applies different image augmentation procedures:

        * random rotations

        * image intensity scaling by multplying the intensity valuse with close to one scale value

        * image scaling, which randomly crops or add pads and scale the image to the original size

    based on the input probability limit probabilistically applies different augmentation cases.

    Args;
        image: input image 4D or 5D tensor
        label: input label 4D or 5D tensor
        prob_lim: probability limit for applying each augmentation case.

    Returns:
        augmented image and l

    See Also:
        * :meth:`img_aug`

    """
    dim = len(image.shape) - 2
    for i in range(image.shape[0]):
        im_, l_ = image[i, ...], label[i, ...]
        if np.random.rand() > prob_lim:  # random rotation
            j = np.floor(np.random.rand() * im_.shape[-2]).astype('int64')
            im_ = np.concatenate((im_[..., j:, :], im_[..., :j, :]), axis=-2)
            l_ = np.concatenate((l_[..., j:, :], l_[..., :j, :]), axis=-2)
        if np.random.rand() > prob_lim:  # random reflection
            j = np.floor(np.random.rand() * im_.shape[-2]).astype('int64')
            im_ = np.concatenate((im_[..., :j:-1, :], im_[..., j::-1, :]), axis=-2)
            l_ = np.concatenate((l_[..., :j:-1, :], l_[..., j::-1, :]), axis=-2)
        if np.random.rand() > prob_lim:  # intensity scaling
            im_ = np.clip(0.1 * (np.random.rand() - 0.5) + im_ * (1 + 0.20 * (np.random.rand() - 0.5)), 0, 1)
        if np.random.rand() > prob_lim:  # image scaling
            scale = 1 + 0.25 * (np.random.rand() - 0.5)
            if scale > 1:
                j = np.zeros(np.ceil(im_.shape[-3] * (scale - 1)).astype('int'), dtype='int')
                im_ = np.concatenate((im_, 0 * im_[..., j, :, :]), axis=-3)
                l_ = np.concatenate((l_, 0 * l_[..., j, :, :]), axis=-3)
            else:
                j = np.ceil(im_.shape[-3] * scale).astype('int64')
                im_ = im_[..., :j, :, :]
                l_ = l_[..., :j, :, :]
            im_ = polar_zoom(im_, image.shape[-3] / im_.shape[-3], order=1)
            l_ = polar_zoom(l_, label.shape[-3] / l_.shape[-3], order=0)
        image[i, ...], label[i, ...] = im_, l_
    return image, label


def polar_zoom(im, scale, order=1):
    """ apply image scaling to polar images along the radius axis.

    Args:
        im: input image
        scale: scaling factor
        order: interpolation order.  0 for nearest and 1 for linear.

    Returns:
        scaled image.
    """
    assert order in [0, 1], "order should be 0 or 1. Got %d" % order
    w = im.shape[-3]
    idx = np.arange(0, np.round(w * scale)) / scale
    a = (idx % 1)[np.newaxis, ..., np.newaxis, np.newaxis]
    if im.ndim != a.ndim:
        a = np.expand_dims(a, axis=0)
    idx = np.clip(np.floor(idx), 0, w - 1).astype(np.int64)
    idx2 = np.clip(idx + 1, 0, w - 1)
    if order == 0:
        idx = np.select([a.squeeze() < 0.5, a.squeeze() >= 0.5], [idx, idx2])
        out = im[..., idx, :, :]
    else:
        out = (1 - a) * im[..., idx, :, :] + a * im[..., idx2, :, :]
    return out


def load_batch(im, datasetID, nBatch, label=None, isAug=False, coord_sys='carts'):
    """ load a batch of data from im and/or label based on dataset (e.g. test).

    This function handel different coordinate system and image augmentation.

    Args:
        im: 4D or 5D image tensor
        datasetID: index of images in im and/or label along the first axis, which belong to this dataset (e.g.  test)
        nBatch: batch size
        label: 4D or 5D label tensor
        isAug: whether to apply data augmentation. See :meth:`img_aug`
        coord_sys: coordinate system {'polar' or 'carts}

    Returns:
         a batch of data as tuple of (image, label)

    See Also:
        * :meth:`load_batch_parallel`

    """

    n = len(datasetID)
    j = np.mod(np.arange(nBatch), n)
    while True:
        j = np.random.randint(0, len(datasetID), nBatch)
        im_ = im[datasetID[j], ...].copy()
        if label is not None:
            label_ = label[datasetID[j], ...].copy()
        else:
            label_ = None
        if isAug:
            im_, label_ = img_aug(im_, label_, coord_sys, 0.5)
        yield (im_, label_)
        j = np.mod(j + nBatch, n)


def load_batch_parallel(im, datasetID, nBatch, label=None, isAug=False, coord_sys='carts', isCritique=False):
    """ load a batch of data from im and/or label based on dataset (e.g. test) using multi-thread.

    This function handel different coordinate system and image augmentation.

    Args:
        im: 4D or 5D image tensor
        datasetID: index of images in im and/or label along the first axis, which belong to this dataset (e.g.  test)
        nBatch: batch size
        label: 4D or 5D label tensor
        isAug: whether to apply data augmentation. See :meth:`img_aug`
        coord_sys: coordinate system {'polar' or 'carts}

    Returns:
        a batch of data as tuple of (image, label)

    See Also:
        * :meth:`load_batch`

    """

    n = len(datasetID)
    j = np.mod(np.arange(nBatch), n)
    while True:
        im_ = im[datasetID[j], ...].copy()
        if label is not None:
            label_ = label[datasetID[j], ...].copy()
        else:
            label_ = None
        if isAug:
            pool = ThreadPool(processes=cpu_count())
            multiple_results = [pool.apply_async(img_aug, (im_[[i], ...], label_[[i], ...], coord_sys, 0.5))
                                for i in range(nBatch)]
            for i, res in enumerate(multiple_results):
                im_[i, ...], label_[i, ...] = res.get()
            pool.close()
        if not isCritique:
            yield (im_, label_)
        else:
            yield (im_, [label_, datasetID[j][:, np.newaxis]])
        j = np.mod(j + nBatch, n)


class LoadBatchGen(Sequence):
    """data generator class, a sub-class of  Keras' Sequence class"""
    def __init__(self, im, datasetID, nBatch, label=None, isAug=False, coord_sys='carts', prob_lim=0.5,
                 isCritique=False):
        self.im, self.label, self.n = im, label, len(datasetID)
        self.prob_lim, self.isCritique = prob_lim, isCritique
        self.datasetID, self.nBatch, self.isAug, self.coord_sys = datasetID, nBatch, isAug, coord_sys
        self.j = np.mod(np.arange(nBatch), self.n)
        
    def __len__(self):
        return int(np.ceil(len(self.datasetID) / self.nBatch))

    def __getitem__(self, item):
        j_ = self.datasetID[self.j].copy()
        im_ = self.im[j_, ...].copy()
        if self.label is not None:
            label_ = self.label[j_, ...].copy()
        else:
            label_ = None
        if self.isAug:
            im_, label_ = img_aug(im_, label_, self.coord_sys, self.prob_lim)
        self.j = np.mod(self.j + self.nBatch, self.n)
        if not self.isCritique:
            return im_, label_
        else:
            return im_, [label_, j_[:, np.newaxis]]


class LoadBatchGenGPU(Sequence):
    """data generator class, a sub-class of  Keras' Sequence class"""
    def __init__(self, im, datasetID, nBatch, label, isAug=True, coord_sys='polar', prob_lim=0.5, isCritique=False,
                 error_list=[]):
        self.prob_lim, self.isCritique, self.error_list = prob_lim, isCritique, np.array(error_list)
        self.W, self.L = im.shape[-3:-1]
        self.datasetID, self.nBatch, self.isAug, self.n = np.array(datasetID), nBatch, isAug, len(datasetID)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.im, self.label = im.astype('float32'), label.astype('float32')
        self.gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        self.graph = tf.Graph()
        self.nSubBatch = nBatch // len(self.gpus)
        self.j = np.mod(np.arange(nBatch), self.n)
        with self.graph.as_default():
            self.sess = tf.Session(config=config)
            self.im_ = tf.placeholder(tf.float32, shape=[None] + list(im.shape[1:]))
            self.label_ = tf.placeholder(tf.float32, shape=[None] + list(label.shape[1:]))
            if self.isAug:
                self.im_, self.label_ = self.polar_aug(self.im_, self.label_)
            print('Data loader is initialized.')

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        out = []
        for i_gpu, gpu in enumerate(self.gpus):
            with tf.device(gpu):
                if self.error_list.size:
                    j_ = np.concatenate((
                        self.datasetID[np.random.randint(0, self.n, self.nSubBatch - self.nSubBatch//2)],
                        self.error_list[np.random.randint(0, len(self.error_list), self.nSubBatch//2)]))
                    print(j_)
                else:
                    j_ = self.datasetID[self.j[(i_gpu * self.nSubBatch):((i_gpu + 1) * self.nSubBatch)]]
                out.append(self.sess.run((self.im_, self.label_), feed_dict={self.im_: self.im[j_, ...],
                                                                             self.label_: self.label[j_, ...]}))
        j_ = self.datasetID[self.j][:, np.newaxis].copy()
        self.j = np.mod(self.j + self.nBatch, self.n)
        if not self.isCritique:
            return (np.concatenate([out[m][0] for m in range(len(self.gpus))], axis=0),
                    np.concatenate([out[m][1] for m in range(len(self.gpus))], axis=0))
        else:
            return (np.concatenate([out[m][0] for m in range(len(self.gpus))], axis=0),
                    [np.concatenate([out[m][1] for m in range(len(self.gpus))], axis=0), j_])

    def polar_aug(self, im__, l__):
        im_out, label_out = tf.zeros([0] + im__.get_shape().as_list()[1:]), \
                                tf.zeros([0] + l__.get_shape().as_list()[1:])
        for i in range(self.nBatch // len(self.gpus)):
            im_, l_ =im__[tf.newaxis, i, ...], l__[tf.newaxis, i, ...]
            if np.random.rand() > self.prob_lim:  # random rotation
                j = np.floor(np.random.rand() * self.L).astype('int64')
                im_ = tf.concat((im_[..., j:, :], im_[..., :j, :]), -2)
                l_ = tf.concat((l_[..., j:, :], l_[..., :j, :]), -2)
            if np.random.rand() > self.prob_lim:  # random reflection
                j = np.floor(np.random.rand() * self.L).astype('int64')
                im_ = tf.concat((im_[..., :j:-1, :], im_[..., j::-1, :]), -2)
                l_ = tf.concat((l_[..., :j:-1, :], l_[..., j::-1, :]), -2)
            if np.random.rand() > self.prob_lim:  # intensity scaling
                im_ = tf.clip_by_value(0.1 * (np.random.rand() - 0.5) +
                                       im_ * (1 + 0.20 * (np.random.rand() - 0.5)), 0, 1)
            if np.random.rand() > self.prob_lim:  # image scaling
                scale = 1 + 0.25 * (np.random.rand() - 0.5)
                idx = scale * np.arange(0, self.W)
                idx0 = np.clip(np.round(idx), 0, self.W - 1).astype(np.int64)
                idx1 = np.clip(np.floor(idx), 0, self.W - 1).astype(np.int64)
                idx2 = np.clip(idx + 1, 0, self.W - 1).astype(np.int64)
                a = tf.constant((idx % 1)[np.newaxis, ..., np.newaxis, np.newaxis], dtype=tf.float32)
                l_ = tf.gather(l_, idx0, axis=-3)
                im_ = (1 - a) * tf.gather(im_, idx1, axis=-3) + a * tf.gather(im_, idx2, axis=-3)
            im_out = tf.concat((im_out, im_), 0)
            label_out = tf.concat((label_out, l_), 0)
        return im_out, label_out

