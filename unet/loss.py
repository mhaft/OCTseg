# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related loss functions"""

import tensorflow as tf
import numpy as np
from scipy.ndimage.morphology import distance_transform_bf


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


def weighted_cross_entropy(label, target):
    """Weighted cross entropy with foreground pixels having ten times higher weights

    Args:
        label: 4D or 5D label tensor
        target: 4D or 5D target tensor

    returns:
        weighted cross entropy value

    """
    # Todo: add positive weight as an argument
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=label, logits=target, pos_weight=10))


def multi_loss_fun(loss_weight):
    """Semantic loss function based on the weighted cross entropy and dice and wighted by the loss weights in the input
    argument

    Args:
        loss_weight: a list with two weights for weighted cross entropy and dice losses, respectively.

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
        elif loss_weight[1] == 0:
            return weighted_cross_entropy(label, target)
        else:
            return loss_weight[0] * weighted_cross_entropy(label, target) + \
                   loss_weight[1] * dice_loss(label, target)

    return multi_loss


def weighted_cross_entropy_with_boundary(label, target):
    """Weighted cross entropy with foreground pixels having ten times higher weights

    Args:
        label: 4D or 5D label tensor
        target: 4D or 5D target tensor

    returns:
        weighted cross entropy value

    """
    # Todo: add positive weight as an argument
    dist = distance_transform_bf(label[:, -1]) + distance_transform_bf(1 - label[:, -1])
    dist = np.logical_and(dist > 5, dist < 20)

    return tf.multiply(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=target), axis=-1),
                     0.001 + 0.010 * label[:, -1] + 0.100 * dist
                     )
