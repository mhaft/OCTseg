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

from unet.unet import unet_model
from unet.ops import accuracy, placeholder_inputs, load_batch
from unet.loss import dice_loss, smooth_loss
from util.load_data import load_train_data

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-exp_def", type=str, default="test", help="experiment definition")
parser.add_argument("-lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("-lr_step", type=float, help="learning rate step for decay", default=1000)
parser.add_argument("-data_path", type=str, default="/home/ubuntu/PSTIFS/", help="data folder path")
parser.add_argument("-nEpoch", type=int, default=3000, help="number of epochs")
parser.add_argument("-nBatch", type=int, default=5, help="batch size")
parser.add_argument("-outCh", type=int, default=2, help="size of output channel")
parser.add_argument("-inCh", type=int, default=1, help="size of input channel")
parser.add_argument("-nZ", type=int, default=1, help="size of input depth")
parser.add_argument("-w", type=int, default=512, help="size of input width")
parser.add_argument("-loss_w", type=str, default="1, 100, 0", help="loss wights")
parser.add_argument("-isAug", type=int, default=1, help="Is data augmentation")


args = parser.parse_args()
experiment_def = args.exp_def
starter_learning_rate = args.lr
folder_path = args.data_path
nEpoch = args.nEpoch
nBatch = args.nBatch
im_shape = (args.nZ, args.w, args.w, args.inCh)
outCh = args.outCh
loss_weight = [float(i) for i in args.loss_w.split(',')]


sess = tf.InteractiveSession()
x, y_ = placeholder_inputs(im_shape, outCh)

y_conv = unet_model(x)

if im_shape[0] == 1:  # 2D
    labels, logits = y_, y_conv
else:  # 3D
    mid_z = (im_shape[0] + 1) // 2
    labels, logits = y_[:, mid_z, ...], y_conv[:, mid_z, ...]
accuracy, jaccard = accuracy(labels, logits)
dice = dice_loss(labels, logits)
smooth = smooth_loss(logits)
cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=10))
loss = loss_weight[0] * cross_entropy + loss_weight[1] * dice + loss_weight[2] * smooth

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(starter_learning_rate, global_step, args.lr_step, 0.1, staircase=True)
train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('Model is initialized.')

im, label, train_data_id, test_data_id, valid_data_id = load_train_data(folder_path, im_shape)
print('Data is loaded')


if not os.path.exists('model/' + experiment_def):
    os.makedirs('model/' + experiment_def)
log_file = 'model/' + experiment_def +'/log.csv'
with open(log_file, 'w') as f:
    f.write('epoch, passed_time_hr, learning_rate, cross_entropy_loss, dice_loss, smooth_loss, Test_JI, Valid_JI, ' +
            str(args) + '\n')
start = time.time()
for epoch in range(nEpoch):
    x1, l1 = load_batch(im, train_data_id, nBatch, label, isAug=args.isAug)
    train_step.run(feed_dict={x: x1, y_: l1})
    if epoch % 100 == 99:
        test_JI, valid_JI= [], []
        for i in range(len(train_data_id) // nBatch):
            x1, l1 = load_batch(im, train_data_id, nBatch, label, iBatch=i)
            test_JI.append(jaccard.eval(feed_dict={x: x1, y_: l1}))
        for i in range(len(valid_data_id) // nBatch):
            j_i = np.arange((i * nBatch), ((i + 1) * nBatch))
            x1, l1 = load_batch(im, valid_data_id, nBatch, label, iBatch=i)
            valid_JI.append(jaccard.eval(feed_dict={x: x1, y_: l1}))
        x1, l1 = load_batch(im, train_data_id, nBatch, label, iBatch=0)
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


# if __name__ == '__main__':
#     pass
