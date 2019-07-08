# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Convert an 2D or 3D image from polar or cylindrical coordinate to the
    cartesian coordinate."""

from __future__ import absolute_import, division, print_function

import time
import numpy as np
import tensorflow as tf

from util.make_data_h5 import make_data_h5

im_shape = (1, 512, 512, 1)
nEpoch = 10
folder_path = 'C:\\Users\\Mohammad\\Desktop\\MachineLearning-3 files\\PSTIFS\\'



def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2_2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


def conv_bn_relu(x, ChIn, ChOut):
    W_conv = weight_variable([3, 3, ChIn, ChOut])
    b_conv = bias_variable([ChOut])
    h_conv = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2d(x, W_conv) + b_conv))
    return h_conv, W_conv, b_conv


def up_conv_2_2(x):
    x_shape = x.get_shape()
    w = weight_variable([2, 2, x_shape[3].value, x_shape[3].value])
    return tf.compat.v1.nn.conv2d_transpose(x, filter=w,
                                  output_shape=[2, 2 * x_shape[1].value,
                                                2 * x_shape[2].value, x_shape[3].value], strides=2)


def img_aug(im, l):
    p_lim = 0.05
    for i in range(im.shape[0]):
        im_, l_ = im[i, ...], l[i, ...]
        if np.random.rand() > p_lim:  # y=x mirror
            im_ = im_.swapaxes(0, 1)
            l_ = l_.swapaxes(0, 1)
        if np.random.rand() > p_lim:  # y mirror
            im_ = im_[:, ::-1, :]
            l_ = l_[:, ::-1]
        if np.random.rand() > p_lim:  # x mirror
            im_ = im_[::-1, :, ]
            l_ = l_[::-1, :]
        if np.random.rand() > p_lim:  # 1st 90 deg rotation
            im_ = np.rot90(im_, k=1, axes=(0, 1))
            l_ = np.rot90(l_, k=1, axes=(0, 1))
        if np.random.rand() > p_lim:  # 2nd 90 degree rotation
            im_ = np.rot90(im_, k=1, axes=(0, 1))
            l_ = np.rot90(l_, k=1, axes=(0, 1))
        if np.random.rand() > p_lim:  # 3rd 90 degree rotation
            im_ = np.rot90(im_, k=1, axes=(0, 1))
            l_ = np.rot90(l_, k=1, axes=(0, 1))
        if np.random.rand() > p_lim:  # salt-and-pepper noise
            im_ = im_ + 0.02 * np.random.rand() - 0.01
        im[i, ...], l[i, ...] = im_, l_
    return im, l


def one_hot(l, num_classes):
    return np.reshape(np.squeeze(np.eye(num_classes)[l.reshape(-1)]), l.shape + (num_classes, ))


sess = tf.compat.v1.InteractiveSession()
x = tf.compat.v1.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], im_shape[3]])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], 3])

# Define the Architecture
h_conv1a, W_conv1a, b_conv1a = conv_bn_relu(x, 1, 32)
h_conv1b, W_conv1b, b_conv1b = conv_bn_relu(h_conv1a, 32, 64)
h_pool1 = max_pool_2_2(h_conv1b)

h_conv2a, W_conv2a, b_conv2a = conv_bn_relu(h_pool1, 64, 64)
h_conv2b, W_conv2b, b_conv2b = conv_bn_relu(h_conv2a, 64, 128)
h_pool2 = max_pool_2_2(h_conv2b)

h_conv3a, W_conv3a, b_conv3a = conv_bn_relu(h_pool2, 128, 128)
h_conv3b, W_conv3b, b_conv3b = conv_bn_relu(h_conv3a, 128, 256)
h_pool3 = max_pool_2_2(h_conv3b)

h_conv4a, W_conv4a, b_conv4a = conv_bn_relu(h_pool3, 256, 256)
h_conv4b, W_conv4b, b_conv4b = conv_bn_relu(h_conv4a, 256, 512)
h_up_conv4 = up_conv_2_2(h_conv4b)

h_conv5a, W_conv5a, b_conv5a = conv_bn_relu(tf.concat([h_conv3b, h_up_conv4], -1),
                                            256 + 512, 256)
h_conv5b, W_conv5b, b_conv5b = conv_bn_relu(h_conv5a, 256, 256)
h_up_conv5 = up_conv_2_2(h_conv5b)

h_conv6a, W_conv6a, b_conv6a = conv_bn_relu(tf.concat([h_conv2b, h_up_conv5], -1),
                                            128 + 256, 128)
h_conv6b, W_conv6b, b_conv6b = conv_bn_relu(h_conv6a, 128, 128)
h_up_conv6 = up_conv_2_2(h_conv6b)

h_conv7a, W_conv7a, b_conv7a = conv_bn_relu(tf.concat([h_conv1b, h_up_conv6], -1),
                                            64 + 128, 64)
h_conv7b, W_conv7b, b_conv7b = conv_bn_relu(h_conv7a, 64, 64)
W_y_conv = weight_variable([1, 1, 64, 3])
y_conv = conv2d(h_conv7b, W_y_conv)


cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-4
lr = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 20000, 0.1, staircase=True)
train_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)

sess.run(tf.compat.v1.initializers.global_variabless())
saver = tf.compat.v1.train.Saver()
print('Model is initialized.')

im, label = make_data_h5(folder_path, im_shape)

im = im.astype(np.float32) / 255
label = np.clip(label, 1, None) - 1
im, label = np.squeeze(im, axis=1), np.squeeze(np.squeeze(label, axis=1), axis=-1)
label = one_hot(label, 3)

train_data_id = np.arange(0, im.shape[0], 2)
test_data_id = np.arange(1, im.shape[0], 4)
valid_data_id = np.arange(3, im.shape[0], 4)
print('Data is loaded')


start = time.time()
for epoch in range(nEpoch):
    j = np.random.randint(0, len(train_data_id), 2)
    x1, l1 = img_aug(im[j, ...], label[j, ...])
    print('epoch {}'.format(epoch))
    train_step.run(feed_dict={x: x1, y_: l1})
    if epoch % 10 == 0:
        print("epoch %d: %f hour to finish. Learning rate: %e. Cross Entropy: %f." % (epoch,
              ((nEpoch - epoch - 1) / (epoch + 1.0) * (time.time() - start) / 3600.0), lr.eval(), cross_entropy.eval(
                  feed_dict={x: x1, y_: l1}),))

    if epoch % 1000 == 999:
        save_path = saver.save(sess, "model-epoch" + str(epoch) + ".ckpt")
        print("epoch %d, Model saved in file: %s" % (epoch, save_path))


