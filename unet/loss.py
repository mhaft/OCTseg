# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related loss functions"""

import tensorflow as tf
from unet.ops import conv_layer


def dice_loss(label, target):
    target = tf.nn.softmax(target)
    yy = tf.multiply(target, target)
    ll = tf.multiply(label, label)
    yl = tf.multiply(target, label)
    return 1 - 2 * (tf.reduce_sum(yl) + 0.5) / (tf.reduce_sum(ll) + tf.reduce_sum(yy) + 1)


def cross_entropy(label, target):
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=label, logits=target, pos_weight=10))
