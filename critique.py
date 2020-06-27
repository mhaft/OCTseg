# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import os
import csv

import h5py
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
from keras import optimizers
import keras.backend as K
from keras.initializers import truncated_normal
from keras.backend.tensorflow_backend import set_session
from keras.losses import get
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import visdom

from util.load_batch import img_aug

# parameters
exp_def = "critique-lr1e-4"
lr = 1e-6
outCh = 4
nBatch = 50
nEpoch = 10000
lastEpoch = 0
dataset = 'Dataset polar Z1-L512-W512-C3.h5'
good_data = 'D:\\MLIntravascularPolarimetry\\MLCardioPullbacks-Batch1-Ver1\\' + dataset
bad_data = 'D:\\MLIntravascularPolarimetry\\MLCardioPullbacks-Batch1-Ver3\\' + dataset


log_file = 'model/' + exp_def + '/log-' + exp_def + '.csv'
if not os.path.exists('model/' + exp_def):
    os.makedirs('model/' + exp_def)
if not lastEpoch or not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write('epoch, Train_Loss, Valid_Loss, Train_acc, Valid_acc, param=' +
                str([lr, outCh, nBatch, nEpoch]) + '\n')


def conv(n, x):
    return KL.Conv2D(n, 3, padding='valid', activation='relu', kernel_initializer=truncated_normal(stddev=0.1))(x)


def cov_pool_layer(n, x, s):
    y = conv(n, x)
    z = conv(n, y)
    return KL.MaxPool2D(pool_size=s)(z)


def critique():
    with tf.name_scope('critique'):
        x = KL.Input(shape=(512, 512, 3))
        y = KL.Input(shape=(512, 512, outCh))
        out = list()
        out.append(KL.Concatenate(-1)([x, y]))
        out.append(cov_pool_layer(32, out[-1], 4))
        out.append(cov_pool_layer(64, out[-1], 4))
        out.append(cov_pool_layer(128, out[-1], 4))
        out.append(KL.Lambda(lambda k: K.reshape(k, (-1, 6 * 6 * 128)))(out[-1]))
        out.append(KL.Dense(1024, activation='relu')(out[-1]))
        out.append(KL.Dense(256, activation='relu')(out[-1]))
        out.append(KL.Dense(1, activation='tanh')(out[-1]))
        return Model(inputs=[x, y], outputs=out[-1], name='critique')


def make_iel_label(label_9class, outCh):
    label = np.zeros(label_9class.shape[:-1] + (outCh,))
    if outCh == 4:
        # 4 channel: Ch1: Lumen , Ch2: visible intima ,  Ch3: visible media ,
        #            Ch0: others ,  note visible is without GW and nonIEL
        nonIEL_GW_mask = np.logical_not(np.logical_or(label_9class[..., 0], label_9class[..., 1]))
        label[..., 1] = label_9class[..., 2]
        label[..., 2] = np.logical_and(label_9class[..., 3], nonIEL_GW_mask)
        label[..., 3] = np.logical_and(label_9class[..., 4], nonIEL_GW_mask)
        label[..., 0] = np.all(np.logical_not(label[..., 1:]), axis=-1)

    elif outCh == 6:
        # 6 channel: Ch1: Lumen  , Ch2: visible intima ,  Ch3: visible media ,
        #            Ch4 : GW outside Lumen ,  Ch5: nonIEL outside Lumen and GW,
        #            Ch0: others ,  note visible is without GW and nonIEL
        nonIEL_GW_mask = np.logical_not(np.logical_or(label_9class[..., 0], label_9class[..., 1]))
        label[..., 1] = label_9class[..., 2]
        label[..., 2] = np.logical_and(label_9class[..., 3], nonIEL_GW_mask)
        label[..., 3] = np.logical_and(label_9class[..., 4], nonIEL_GW_mask)
        outside_mask = np.all(np.logical_not(label[..., 1:4]), axis=-1)
        label[..., 4] = np.logical_and(label_9class[..., 0], outside_mask)
        nonIEL_withoutGW = np.logical_and(label_9class[..., 1], np.logical_not(label[..., 0]))
        label[..., 5] = np.logical_and(nonIEL_withoutGW, outside_mask)
        label[..., 0] = np.all(np.logical_not(label[..., 1:]), axis=-1)
    return label


def loss(y_, y):
    # return - tf.reduce_mean(y * (1 + y_)) \
    #        + tf.reduce_mean(y * (1 - y_)) \
    #        + 10 * tf.reduce_mean(tf.square(y))  # +  0.01 * gradient_penalty
    return tf.reduce_mean(tf.abs(1 - y_ * y))


# GPU settings
# os.environ["CUDA_VISIBLE_DEVICES"] = " "
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))
numGPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

model_ = critique()
vis = visdom.Visdom(env='main')

if lastEpoch:
    model_.load_weights('model/' + exp_def + '/model-epoch%06d.h5' % lastEpoch)
    with open(log_file, 'r') as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        _ = next(reader)  # header
        data = []
        for row in reader:
            data.append([float(i) for i in row])
        data = np.array(data)
    vis.line(data[:, 1:3], data[:, [0, 0]], win=(exp_def + '_loss'),
             opts=dict(title=exp_def, xlabel='Epoch', ylabel='Loss', legend=['Training Loss', 'Validation Loss']))
    vis.line(data[:, 3:5], data[:, [0, 0]], win=(exp_def + '_acc'),
             opts=dict(title=exp_def, xlabel='Epoch', ylabel='Acc', legend=['Training Acc', 'Validation Acc']))

with h5py.File(good_data, 'r') as f:
    im, label_good, train_data_id, valid_data_id = np.array(f.get('/im')), np.array(f.get('/label')), \
                                                   np.array(f.get('/train_data_id')), np.array(f.get('/valid_data_id'))
with h5py.File(bad_data, 'r') as f:
    label_bad = np.array(f.get('/label'))


label_good, label_bad = make_iel_label(label_good), make_iel_label(label_bad)
out_train = np.concatenate((np.ones((train_data_id.shape[0], 1)), -1 * np.ones((train_data_id.shape[0], 1))), axis=0)
out_valid = np.concatenate((np.ones((valid_data_id.shape[0], 1)), -1 * np.ones((valid_data_id.shape[0], 1))), axis=0)
im_train = np.tile(im[train_data_id, ...], (2, 1, 1, 1))
im_valid = np.tile(im[valid_data_id, ...], (2, 1, 1, 1))
label_train = np.concatenate((label_good[train_data_id, ...], label_bad[train_data_id, ...]), axis=0)
label_valid = np.concatenate((label_good[valid_data_id, ...], label_bad[valid_data_id, ...]), axis=0)
print('Data is loaded. Training: %d, validation: %d' % (label_train.shape[0], label_valid.shape[0]))


gradients = tf.concat(tf.gradients(model_.output, model_.input), axis=-1)
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[-1]))
gradient_penalty = tf.reduce_mean((slopes-1)**2)


if numGPU > 1:
    model = multi_gpu_model(model_, gpus=numGPU)
else:
    model = model_
model.compile(loss=get(loss), optimizer=optimizers.rmsprop(lr=lr, clipnorm=1.0, clipvalue=1.0))  #


# i1 = [0, im_train.shape[0] // 2]
# im_train, label_train, out_train = im_train[i1, ...], label_train[i1, ...], out_train[i1, ...]
# i1 = [0, im_valid.shape[0] // 2]
# im_valid, label_valid, out_valid = im_valid[i1, ...], label_valid[i1, ...], out_valid[i1, ...]

for iEpoch in range(lastEpoch, nEpoch):
    print('Epoch %d / %d' % (iEpoch + 1, nEpoch))
    his = model.fit(x=list(img_aug(im_train.copy(), label_train.copy(), 'polar', 1 - iEpoch / nEpoch)),
                    y=out_train, batch_size=nBatch, epochs=1, shuffle=True,
                    validation_data=([im_valid, label_valid], out_valid), verbose=1)
    loss_t = model.evaluate(x=[im_train, label_train], y=out_train, verbose=0, batch_size=nBatch)
    loss_v = model.evaluate(x=[im_valid, label_valid], y=out_valid, verbose=0, batch_size=nBatch)
    vis.line(np.array([[loss_t], [loss_v]]).T, np.array([[iEpoch + 1], [iEpoch + 1]]).T, win=(exp_def + '_loss'),
             update='append' if iEpoch > 0 else 'replace', opts=dict(title=exp_def, xlabel='Epoch', ylabel='Loss',
                                                                     legend=['Training Loss', 'Validation Loss']))
    u0t = model.predict(x=[im_train, label_train], batch_size=nBatch)
    u0v = model.predict(x=[im_valid, label_valid], batch_size=nBatch)
    j = 0
    acc_t = (np.mean(u0t[out_train == 1] >= j) + np.mean(u0t[out_train == -1] < j)) / 2
    j = 0
    acc_v = (np.mean(u0v[out_valid == 1] >= j) + np.mean(u0v[out_valid == -1] < j)) / 2
    vis.line(np.array([[acc_t], [acc_v]]).T, np.array([[iEpoch + 1], [iEpoch + 1]]).T, win=(exp_def + '_acc'),
             update='append' if iEpoch > 0 else 'replace', opts=dict(title=exp_def, xlabel='Epoch', ylabel='Acc',
                                                                     legend=['Training Acc', 'Validation Acc']))
    smi = '<p style="color:blue;font-family:monospace;font-size:80%;">' + \
          '<br>'.join(os.popen('nvidia-smi').read().split('\n')[3:13]) + '</p>'
    vis.text(smi, win='nvidia-smi')
    with open(log_file, 'a') as f:
        f.write("%d, %f, %f, %f, %f \n" % (iEpoch + 1, loss_t, loss_v, acc_t, acc_v))

    if iEpoch % 100 == 99:
        model_.save('model/' + exp_def + '/model-epoch%06d.h5' % (iEpoch + 1))

    vis.scatter(np.concatenate((np.arange(u0t.size)[..., np.newaxis],
                                np.concatenate((u0t[out_train == 1],
                                                u0t[out_train == -1]))[..., np.newaxis]), axis=1),
                win=(exp_def + '_prediction1'),
                opts=dict(title='train prediction', xlabel='sample', ylabel='value', markersize=5))
    vis.scatter(np.concatenate((np.arange(u0v.size)[..., np.newaxis],
                                np.concatenate((u0v[out_valid == 1],
                                                u0v[out_valid == -1]))[..., np.newaxis]), axis=1),
                win=(exp_def + '_prediction2'),
                opts = dict(title='valid prediction', xlabel='sample', ylabel='value', markersize=5))
    # plt.plot(u0t[out_train == 1] - u0t[out_train == -1], 'r*')
    # plt.plot(u0t, 'r*')
    # plt.show()


# clipvalue
0.5
# clipnorm
0.7619047619047619
0.7857142857142857
# clipnorm with weight regularizer
0.8809523809523809
# nothing
0.7142857142857143

