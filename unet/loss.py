# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related loss functions"""

import tensorflow as tf
import numpy as np


def dice_loss(label, target):
    """Soft Dice coefficient loss

    TP, FP, and FN are true positive, false positive, and false negative.

    .. math::
        dice  &=  \\frac{2 \\times TP}{ 2 \\times TP + FN + FP} \\\\
        dice  &=  \\frac{2 \\times TP}{(TP + FN) + (TP + FP)}

    objective is to maximize the dice, thus the loss is negate of dice for numerical stability (+1 in denominator)
    and fixing the loss range (+1 in numerator and +1 to the negated dice).

    The final Dice loss is formulated as

    .. math:: dice \ loss = 1 - \\frac{2 \\times TP + 1}{(TP + FN) + (TP + FP ) + 1}

    it is soft as each components of the confusion matrix (TP, FP, and FN) are estimated by dot product of
    probability instead of hard classification

    Args:
        label: 4D or 5D label tensor
        target: 4D or 5D target tensor

    Returns:
         dice loss

    """
    eps = 1e-6
    target = tf.clip_by_value(tf.nn.softmax(target), eps, 1 - eps)
    target, label = target[..., 1:], label[..., 1:]
    yy = tf.multiply(target, target)
    ll = tf.multiply(label, label)
    yl = tf.multiply(target, label)
    axis_ = tf.range(1, tf.rank(label) - 1)
    return tf.reduce_mean(1 - (2 * tf.reduce_sum(yl, axis=axis_, keepdims=True)) /
                              (tf.reduce_sum(ll, axis=axis_, keepdims=True) +
                               tf.reduce_sum(yy, axis=axis_, keepdims=True) + eps), axis=-1)


def weighted_cross_entropy_fun(loss_weight):
    """Weighted cross entropy with foreground pixels having ten times higher weights

    Args:
        label: 4D or 5D label tensor
        target: 4D or 5D target tensor
        loss_weight: pos_weight for the :meth:`tf.nn.weighted_cross_entropy_with_logits`

    returns:
        weighted cross entropy value

    """
    def weighted_cross_entropy(label, target):
        return tf.nn.weighted_cross_entropy_with_logits(labels=label, logits=target, pos_weight=loss_weight)
    return weighted_cross_entropy


def multi_loss(loss_weight, numClass):
    """Semantic loss function based on the weighted cross entropy and dice and wighted by the loss weights in the input
    argument

    Args:
        loss_weight: a list with three weights for weighted cross entropy, foreground, and dice losses, respectively.

    Returns:
         function, which similar to :meth:`weighted_cross_entropy` and :meth:`dice_loss`
                has label and target arguments

    See Also:
        * :meth:`weighted_cross_entropy`
        * :meth:`dice_loss`

    """
    loss_weight = np.array(loss_weight)

    def multi_loss_(label, target):
        shape = label.get_shape()
        # just evaluate at the center slice in 3D inputs
        if len(shape) == 5 and shape[1].value is not None:
            i = shape[1].value // 2
            label, target = label[:, i, ...], target[:, i, ...]
        with tf.name_scope('multi_loss'):
            if loss_weight.size < numClass:
                print('Loss weight size is less than the number of class. Got %d and %d. Dice loss is selected.' %
                      (loss_weight.size, numClass))
                return dice_loss(label, target)
            elif loss_weight.size == numClass:
                return weighted_categorical_crossentropy(loss_weight)(label, target)
            elif loss_weight.size == numClass + 1:
                return weighted_cross_entropy_with_boundary(loss_weight)(label, target)
            elif loss_weight.size == numClass + 2:
                return weighted_cross_entropy_with_boundary(loss_weight[:-1])(label, target) + \
                       loss_weight[-1] * dice_loss(label, target)
            elif loss_weight.size == numClass + 3:
                return weighted_cross_entropy_with_boundary(loss_weight[:-2])(label, target) + \
                       loss_weight[-2] * dice_loss(label, target) + \
                       loss_weight[-1] * boundary_transition_loss()(label, target)

            else:
                raise Exception('Loss weight has too many elements. Got %d' % loss_weight.size)

    return multi_loss_


def weighted_cross_entropy_with_boundary(loss_weight, boundary_r=10):
    """Weighted cross entropy with foreground and boundaries pixels having  higher weights

     Args:
         loss_weight: a list with of weights with length equal to the total number of classes plus one. The last
                      value is the loss weight fot the boundary and other elements are classes weights
         label: 4D or 5D label tensor
         target: 4D or 5D target tensor
         boundary_r: radius of boundary considered for the higher loss values

     returns:
         weighted cross entropy value

     See Also:
         * :meth:`mask_boundary_neighborhood`

     """

    loss_weight = np.array(loss_weight).astype('float32')

    def weighted_cross_entropy_with_boundary_(label, target):
        with tf.name_scope('wCEb'):
            dist_mask = mask_boundary_neighborhood(label, r=boundary_r, numClass=(loss_weight.size - 1))
            eps = 1e-6
            target_ = tf.clip_by_value(tf.nn.softmax(target), eps, 1 - eps)
            cross_entropy = tf.reduce_mean(- label * tf.log(target_), axis=-1)
        return (tf.multiply(cross_entropy, dist_mask) * loss_weight[-1]) + \
                weighted_categorical_crossentropy(loss_weight[:-1])(label, target)

    return weighted_cross_entropy_with_boundary_


def mask_boundary_neighborhood(label, r=5, numClass=2):
    """ mask the neighborhood of the boundary between foreground and background.

    Args:
        label: input label_
        r: neighborhood radius

    Returns:
        mask

    See Also:
         * :meth:`weighted_cross_entropy_with_boundary`

    """

    def mask_boundary_neighborhood_(label_, class_id):
        fg = label_[..., class_id:(class_id + 1)] > 0
        if tf.rank(fg) == 3:
            tf.expand_dims(fg, axis=-1)
        bg = tf.logical_not(fg)
        before = tf.logical_xor(tf.nn.conv2d(tf.cast(bg, tf.float32),
                                             filter=tf.ones([r, 1, 1, 1]), padding="SAME") > 0, bg)
        after = tf.logical_xor(tf.nn.conv2d(tf.cast(fg, tf.float32),
                                            filter=tf.ones([r, 1, 1, 1]), padding="SAME") > 0, fg)
        return tf.logical_or(before, after)

    out = mask_boundary_neighborhood_(label, 1)
    out = tf.logical_or(out, mask_boundary_neighborhood_(label, 2))
    out = tf.logical_or(out, mask_boundary_neighborhood_(label, 3))
    out = tf.cast(out[..., 0], tf.float32)
    out = out * tf.cast(tf.size(out), dtype=tf.float32) / tf.reduce_sum(out)  # normalize
    return out


def weighted_categorical_crossentropy(loss_weight):
    """ weighted categorical crossentropy

    Args:
         loss_weight: a list with three weights for all pixels outside the mask, foreground, and pixels close to the
                        boundary, respectively.
    """
    loss_weight = tf.Variable(np.array(loss_weight).astype('float32'))

    def weighted_categorical_crossentropy_(label, target):
        eps = 1e-6
        target = tf.clip_by_value(tf.nn.softmax(target), eps, 1 - eps)
        loss_ = - ((label * tf.log(target))
                   * loss_weight * tf.cast(tf.size(label[..., 0]), dtype=tf.float32) /
                   (tf.reduce_sum(label, axis=tf.range(tf.rank(label) - 1)) + 1))
        return tf.reduce_mean(loss_, axis=-1)

    return weighted_categorical_crossentropy_


def boundary_transition_loss(isPixel = False):
    """
    Compare the number of boundaries along the columns. The value can be assinged to all the pixels in the A-line or
    can be assigned to the pixels at the boundary.

    """
    def boundary_transition_loss_(label, target):
        eps = 1e-6

        def num_boundary(b):
            b = 1 + tf.tanh((100 / eps) * (b - tf.reduce_max(b, axis=-1, keepdims=True)))
            return 0.5 * tf.reduce_sum(tf.reduce_sum(tf.abs(b[..., 1:, :, :] - b[..., :-1, :, :]),
                                                     axis=-1), axis=-2, keepdims=True)

        # apply the loss just to the boundary instead of the whole column
        def num_boundary_pixel(b, keepdims):
            seg_prob_sat = tf.tanh((100 / eps) * (b - tf.reduce_max(b, axis=-1, keepdims=True)))
            b = 1 - 0.5 * tf.reduce_sum(tf.abs(seg_prob_sat[..., 1:, :, :] - seg_prob_sat[..., :-1, :, :]), axis=-1)
            b_c = tf.concat([b[..., :1, :] * 0, 1 - b[..., :-1, :] * b[..., 1:, :], b[..., :1, :] * 0], -2)
            return b_c if keepdims else tf.reduce_sum(b_c, axis=-2, keepdims=True)

        target = tf.clip_by_value(tf.nn.softmax(target), eps, 1 - eps)
        if isPixel:
            return tf.abs(num_boundary_pixel(label, False) - num_boundary_pixel(target, False)) * \
                   num_boundary_pixel(target, True)
        else:
            tf.abs(num_boundary(label) - num_boundary(target))

    return boundary_transition_loss_


def new_loss2x3ch(loss_weight):
    """Loss function for a particular label definition, which has two separate set of labels and each set has
         three classes. One is for wall structures and one is for artifacts. For the dataset compiled in 2020,
        weights are [0.02, 0.18, 0.80, 0.09, 0.75, 0.16,  0.1].
    """

    def loss_(label, target):
        target1, target2 = target[..., :3], target[..., 3:]
        label1, label2, mask1 = label[..., :3], label[..., 3:], label[..., 3]
        mask2 = tf.cast(tf.logical_and(
                        tf.reduce_sum(label[..., 1], axis=-3, keepdims=True) <= 34,
                        tf.reduce_sum(label[..., 2], axis=-3, keepdims=True) <= 19), tf.float32)
        L1 = multi_loss(loss_weight[[0, 1, 2, 6, 7, 8]], 3)(label1, target1)
        L2 = multi_loss(loss_weight[[3, 4, 5]], 3)(label2, target2)
        return ((mask1 * mask2 + loss_weight[9] * mask1 * (1 - mask2)
                 + loss_weight[10] * (1 - mask1)) * L1 + loss_weight[11] * L2)

    return loss_
