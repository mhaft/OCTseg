# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""CNN related loss functions"""

import tensorflow as tf
from unet.ops import conv


def dice_loss(target, label):
    target = tf.nn.softmax(target)
    yy = tf.multiply(target, target)
    ll = tf.multiply(label, label)
    yl = tf.multiply(target, label)
    return 1 - 2 * (tf.reduce_sum(yl) + 0.5) / (tf.reduce_sum(ll) + tf.reduce_sum(yy) + 1)


def smooth_loss(target):
    y = tf.nn.softmax(target)
    w = tf.ones((3, 3, y.shape[-1], y.shape[-1]))
    y_smooth = conv(y, w) / tf.cast(tf.size(w), tf.float32)
    return tf.losses.mean_squared_error(y_smooth, y)