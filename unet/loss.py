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

    def weighted_cross_entropy_with_boundary_(label, target):
        with tf.name_scope('wCEb'):
            dist_mask = mask_boundary_neighborhood(label, r=boundary_r, numClass=(loss_weight.size - 1))
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=target)
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
        loss_ = - ((label * tf.log(target) + (1 - label) * tf.log(1 - target))
                   * loss_weight * tf.cast(tf.size(label), dtype=tf.float32) /
                   (tf.reduce_sum(label, axis=tf.range(tf.rank(label) - 1)) + 1))
        return tf.reduce_sum(loss_, axis=-1)

    return weighted_categorical_crossentropy_