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
import tifffile
import argparse
import os
import numpy as np
import tensorflow as tf

from util.make_data_h5 import make_data_h5

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-exp_def", type=str, default="test", help="experiment definition")
parser.add_argument("-lr", type=float, help="learning rate", default=1e-5)
parser.add_argument("-data_path", type=str, default="./data/", help="data folder path")
parser.add_argument("-nEpoch", type=int, default=1000, help="number if epochs")
parser.add_argument("-nBatch", type=int, default=1, help="batch size")
parser.add_argument("-outCh", type=int, default=2, help="size of output channel")
parser.add_argument("-inCh", type=int, default=1, help="size of input channel")
parser.add_argument("-nZ", type=int, default=1, help="size of input depth")
parser.add_argument("-w", type=int, default=512, help="size of input width")
parser.add_argument("-loss_w", type=str, default="1, 0.5, 0", help="loss wights")

args = parser.parse_args()
experiment_def = args.exp_def
starter_learning_rate = args.lr
folder_path = args.data_path
nEpoch = args.nEpoch
nBatch = args.nBatch
im_shape = (args.nZ, args.w, args.w, args.inCh)
loss_weight = [float(i) for i in args.loss_w.split(',')]

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
    # h_conv = tf.nn.leaky_relu(tf.keras.layers.BatchNormalization()(conv2d(x, W_conv) + b_conv))
    h_conv = tf.nn.leaky_relu(conv2d(x, W_conv) + b_conv)
    return h_conv, W_conv, b_conv


def up_conv_2_2(x):
    x_shape = x.get_shape()
    # w = weight_variable([2, 2, x_shape[3].value, x_shape[3].value])
    # return tf.nn.conv2d_transpose(x, filter=w, output_shape=[2, 2 * x_shape[1].value,
    #                                             2 * x_shape[2].value, x_shape[3].value], strides=2)
    return tf.keras.layers.Conv2DTranspose(x_shape[3].value, 2, 2)(x)


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
        # if np.random.rand() > p_lim:  # salt-and-pepper noise
        #     im_ = im_ + 0.01 * (np.random.rand() - 0.5)
        im[i, ...], l[i, ...] = im_, l_
    return im, l


def one_hot(l, num_classes):
    return np.reshape(np.squeeze(np.eye(num_classes)[l.reshape(-1)]), l.shape + (num_classes, ))


def dice_loss(target, label):
    target = tf.nn.softmax(target)
    yy = tf.multiply(target, target)
    ll = tf.multiply(label, label)
    yl = tf.multiply(target, label)
    return 1 - 2 * (tf.reduce_sum(yl) + 0.5) / (tf.reduce_sum(ll) + tf.reduce_sum(yy) + 1)


def smooth_loss(target):
    y = tf.nn.softmax(target)
    w = tf.ones((3, 3, y.shape[-1], y.shape[-1]))
    y_smooth = conv2d(y, w) / tf.cast(tf.size(w), tf.float32)
    return tf.losses.mean_squared_error(y_smooth, y)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], im_shape[3]])
y_ = tf.placeholder(tf.float32, shape=[None, im_shape[1], im_shape[2], 2])

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
W_y_conv = weight_variable([1, 1, 64, 2])
y_conv = conv2d(h_conv7b, W_y_conv)

dice = dice_loss(y_conv, y_)
smooth = smooth_loss(y_conv)
cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y_, logits=y_conv, pos_weight=10))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, -1), tf.argmax(y_, -1)), tf.float32), [1, 2])
allButTN = tf.maximum(tf.argmax(y_conv, -1), tf.argmax(y_, -1))
correct_prediction = tf.multiply(tf.argmax(y_conv, -1), tf.argmax(y_, -1))
jaccard = tf.divide(tf.reduce_sum(tf.cast(correct_prediction, tf.float32)),
                    tf.reduce_sum(tf.cast(allButTN, tf.float32)))

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.1, staircase=True)
loss = loss_weight[0] * cross_entropy + loss_weight[1] * dice + loss_weight[2] * smooth
train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('Model is initialized.')

im, label = make_data_h5(folder_path, im_shape)
assert len(im) > 0, "The data folder is empty: %s"%folder_path

im = im.astype(np.float32) / 255
# label = np.clip(label, 1, None) - 1
label = (label == 3).astype(np.uint8)
im, label = np.squeeze(im, axis=1), np.squeeze(np.squeeze(label, axis=1), axis=-1)
label = one_hot(label, 2)

train_data_id = np.arange(0, im.shape[0], 2)
test_data_id = np.arange(1, im.shape[0], 4)
valid_data_id = np.arange(3, im.shape[0], 4)
print('Data is loaded')


if not os.path.exists('model/' + experiment_def):
    os.makedirs('model/' + experiment_def)
log_file = 'model/' + experiment_def +'/log.csv'
with open(log_file, 'w') as f:
    f.write('epoch, passed_time_hr, learning_rate, cross_entropy_loss, dice_loss, smooth_loss, Test_JI, Valid_JI \n')
start = time.time()
for epoch in range(nEpoch):
    j = np.random.randint(0, len(train_data_id), nBatch)
    x1, l1 = img_aug(im[train_data_id[j], ...], label[train_data_id[j], ...])
    train_step.run(feed_dict={x: x1, y_: l1})
    if epoch % 100 == 99:
        test_JI = []
        for i in range(len(train_data_id) // nBatch):
            j1, j2 = (i * nBatch), ((i + 1) * nBatch)
            x1, l1 = img_aug(im[train_data_id[j1:j2], ...], label[train_data_id[j1:j2], ...])
            test_JI.append(jaccard.eval(feed_dict={x: x1, y_: l1}))
        valid_JI = []
        for i in range(len(valid_data_id) // nBatch):
            j_i = np.arange((i * nBatch), ((i + 1) * nBatch))
            x1, l1 = img_aug(im[valid_data_id[j_i], ...], label[valid_data_id[j_i], ...])
            valid_JI.append(jaccard.eval(feed_dict={x: x1, y_: l1}))
        x1, l1 = im[train_data_id[0:nBatch], ...], label[train_data_id[0:nBatch], ...]
        log_value = (epoch + 1, (nEpoch - epoch - 1) / (epoch + 1.0) * (time.time() - start) / 3600.0,
                     lr.eval(), cross_entropy.eval(feed_dict={x: x1, y_: l1}),
                     dice.eval(feed_dict={x: x1, y_: l1}), smooth.eval(feed_dict={x: x1, y_: l1}),
                     np.mean(test_JI), np.mean(valid_JI))
        print("epoch %d: %f hour to finish. Learning rate: %e. Cross entropy: %f. Dice loss: %f. Smooth_loss: %f. "
              "Test JI: %f. Valid JI: %f." % log_value)
        with open(log_file, 'a') as f:
            f.write("%d, %f, %e, %f, %f, %f, %f, %f \n" % log_value)
    if epoch % 1000 == 999:
        save_path = saver.save(sess, 'model/' + experiment_def + '/model-epoch' + str(epoch + 1) + '.ckpt')
        print("epoch %d, Model saved in file: %s" % (epoch + 1, save_path))

label = np.argmax(label, -1)
out = np.zeros_like(label)
for i in range(im.shape[0] // nBatch + 1):
    x1 = im[(i * nBatch):((i + 1) * nBatch), ...]
    out[(i * nBatch):((i + 1) * nBatch), ...] = np.argmax(y_conv.eval(feed_dict={x: x1}), -1)

tifffile.imwrite('model/' + experiment_def + '/label.tif', label.astype(np.uint8))
tifffile.imwrite('model/' + experiment_def + '/out.tif', out.astype(np.uint8))
tifffile.imwrite('model/' + experiment_def + '/im.tif', (im * 255).astype(np.uint8).squeeze())
