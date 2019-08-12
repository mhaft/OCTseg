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

import numpy as np
from scipy.ndimage import zoom


def img_rand_scale(im, scale, order):
    """Scale one image or label batch in Cartesian coordinate system.

    scale the image based on the input scale value and interpolation order followed by cropping or padding to
    maintain the original image shape.  For interpolation close to the boundaries,  the reflection mode is used.

    :param im: 3D or 4D image or label tensor
    :param scale: scalar scale values for x and y direction
    :param order: interpolation order
    :return: same size image with the scale image in the center of it.
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


def img_aug(im, l, coord_sys, p_lim=0.5):
    """Image augmentation manager.

    Based on the coordinate system (*Polar* vs.  *Cartesian*),  it selects the corresponding method.

    :param im: input image 4D or 5D tensor
    :param l: input label 4D or 5D tensor
    :param coord_sys: coordinate system.  'polar' or 'carts' for Polar and Cartesian,  respectively.
    :param p_lim: probability limit for applying each augmentation case.
    :return: augmented im and l

    :seemore: :meth:`img_aug_carts`, :meth:`img_aug_polar`
    """
    assert coord_sys in ['carts', 'polar'], 'the coord_sys should be carts or polar. got %d' % coord_sys
    if coord_sys == 'carts':
        return img_aug_carts(im, l, p_lim)
    else:
        return img_aug_polar(im, l, p_lim)


def img_aug_carts(im, l, p_lim=0.5):
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

    :param im: input image 4D or 5D tensor
    :param l: input label 4D or 5D tensor
    :param p_lim: probability limit for applying each augmentation case.
    :return: augmented im and l

    :seemore: :meth:`img_aug`
    """
    dim = len(im.shape) - 2
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


def img_aug_polar(im, l, p_lim=0.5):
    """Data augmentation in Polar coordinate.

    Applies different image augmentation procedures:
        * random rotations

        * image intensity scaling by multplying the intensity valuse with close to one scale value

        * image scaling, which randomly crops or add pads and scale the image to the original size

    based on the input probability limit probabilistically applies different augmentation cases.

    :param im: input image 4D or 5D tensor
    :param l: input label 4D or 5D tensor
    :param p_lim: probability limit for applying each augmentation case.
    :return: augmented im and l

    :seemore: :meth:`img_aug`
    """
    dim = len(im.shape) - 2
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


def load_batch(im, datasetID, nBatch, label=None, isAug=False, coord_sys='carts'):
    """ load a batch of data from im and/or label based on dataset (e.g. test).

    This function handel different coordinate system and image augmentation.

    :param im: 4D or 5D image tensor
    :param datasetID: index of images in im and/or label along the first axis, which belong to this dataset (e.g.  test)
    :param nBatch: batch size
    :param label: 4D or 5D label tensor
    :param isAug: whether to apply data augmentation. See :meth:`img_aug`
    :param coord_sys: coordinate system {'polar' or 'carts}
    :return: a batch of data as tuple of (image, label)

    :seemore: :meth:`load_batch_parallel`
    """

    while True:
        j = np.random.randint(0, len(datasetID), nBatch)
        im_ = im[datasetID[j], ...]
        if label is not None:
            label_ = label[datasetID[j], ...]
        else:
            label_ = None
        if isAug:
            im_, label_ = img_aug(im_, label_, coord_sys, 0.5)
        yield (im_, label_)


def load_batch_parallel(im, datasetID, nBatch, label=None, isAug=False, coord_sys='carts'):
    """ load a batch of data from im and/or label based on dataset (e.g. test) using multi-thread.

    This function handel different coordinate system and image augmentation.

    :param im: 4D or 5D image tensor
    :param datasetID: index of images in im and/or label along the first axis, which belong to this dataset (e.g.  test)
    :param nBatch: batch size
    :param label: 4D or 5D label tensor
    :param isAug: whether to apply data augmentation. See :meth:`img_aug`
    :param coord_sys: coordinate system {'polar' or 'carts}
    :return: a batch of data as tuple of (image, label)

    :seemore: :meth:`load_batch`
    """

    while True:
        j = np.random.randint(0, len(datasetID), nBatch)
        im_ = im[datasetID[j], ...]
        if label is not None:
            label_ = label[datasetID[j], ...]
        else:
            label_ = None
        if isAug:
            pool = ThreadPool(processes=cpu_count())
            multiple_results = [pool.apply_async(img_aug, (im_[[i], ...], label_[[i], ...], coord_sys, 0.5))
                                for i in range(nBatch)]
            for i, res in enumerate(multiple_results):
                im_[i, ...], label_[i, ...] = res.get()
            pool.close()
        yield (im_, label_)
