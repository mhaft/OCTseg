# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related loss functions"""

import tensorflow as tf


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
    target = tf.nn.softmax(target)
    target, label = target[..., 1:], label[..., 1:]
    yy = tf.multiply(target, target)
    ll = tf.multiply(label, label)
    yl = tf.multiply(target, label)
    #
    return 1 - (2 * tf.reduce_sum(yl) + 1) / (tf.reduce_sum(ll) + tf.reduce_sum(yy) + 1)


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
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=label, logits=target,
                                                                       pos_weight=loss_weight))
    return weighted_cross_entropy


def multi_loss_fun(loss_weight):
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

    def multi_loss(label, target):
        shape = label.get_shape()
        if len(shape) == 5 and shape[1].value is not None:
            i = shape[1].value // 2
            label, target = label[:, i, ...], target[:, i, ...]
        if loss_weight[0] == 0:
            return dice_loss(label, target)
        elif loss_weight[2] == 0:
            return weighted_cross_entropy_fun(loss_weight[1])(label, target)
        else:
            return loss_weight[0] * weighted_cross_entropy_fun(loss_weight[1])(label, target) + \
                   loss_weight[2] * dice_loss(label, target)

    return multi_loss


def weighted_cross_entropy_with_boundary_fun(loss_weight):
    """Weighted cross entropy with foreground pixels having ten times higher weights

     Args:
         loss_weight: a list with three weights for all pixels outside the mask, foreground, and pixels close to the
                        boundary, respectively.
         label: 4D or 5D label tensor
         target: 4D or 5D target tensor

     returns:
         weighted cross entropy value

     See Also:
         * :meth:`mask_boundary_neighborhood`

     """

    def weighted_cross_entropy_with_boundary(label, target):
        dist_mask = mask_boundary_neighborhood(label, r=10)
        dist_mask = dist_mask * tf.cast(tf.size(dist_mask), dtype=tf.float32) / tf.reduce_sum(dist_mask)
        mask = tf.cast(tf.logical_not(tf.reduce_any(tf.math.equal(label[..., 1:], 2), axis=-1, keepdims=True)),
                       dtype=tf.float32)
        label2 = tf.multiply(label, mask)
        target2 = tf.multiply(target, mask)
        mask = mask * tf.cast(tf.size(mask), dtype=tf.float32) / tf.reduce_sum(mask)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label2, logits=target2)
        loss = (loss_weight[0] +
                loss_weight[1] * tf.cast(tf.reduce_any(tf.math.equal(label[..., 1:], 1), axis=-1), dtype=tf.float32) +
                loss_weight[2] * dist_mask[..., -1])
        return tf.multiply(cross_entropy, tf.multiply(mask[..., 0], loss))

    return weighted_cross_entropy_with_boundary


def mask_boundary_neighborhood(label, r=5):
    """ mask the neighborhood of the boundary between foreground and background.

    Args:
        label: input label
        r: neighborhood radius

    Returns:
        mask

    See Also:
         * :meth:`weighted_cross_entropy_with_boundary`

    """
    fg = tf.reduce_any(tf.math.equal(label[..., 1:], 1), axis=-1, keepdims=True)
    if tf.rank(fg) == 3:
        tf.expand_dims(fg, axis=-1)
    bg = tf.logical_not(fg)
    before = tf.logical_xor(tf.nn.conv2d(tf.cast(bg, tf.float32), filter=tf.ones([r, r, 1, 1]), padding="SAME") > 0, bg)
    after = tf.logical_xor(tf.nn.conv2d(tf.cast(fg, tf.float32), filter=tf.ones([r, r, 1, 1]), padding="SAME") > 0, fg)
    return tf.cast(tf.logical_or(before, after), tf.float32)
