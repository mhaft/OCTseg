# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import os
import csv
import time

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

from util.load_batch import img_aug

# parameters
exp_def = "critique-outCh6_v6"
lr = 1e-6
outCh = 6
nBatch = 30
nEpoch = 1000
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
    return tf.reduce_mean(- y_ * y) + 10 * tf.reduce_mean(tf.square(1 - gradients))


# GPU settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))
numGPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

model_ = critique()

if lastEpoch:
    model_.load_weights('model/' + exp_def + '/model-epoch%06d.h5' % lastEpoch)
    print('model %d  is loaded.' % lastEpoch)
    with open(log_file, 'r') as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        _ = next(reader)  # header
        data = []
        for row in reader:
            data.append([float(i) for i in row])
        data = np.array(data)

with h5py.File(good_data, 'r') as f:
    im, label_good, train_data_id, valid_data_id = np.array(f.get('/im')), np.array(f.get('/label')), \
                                                   np.array(f.get('/train_data_id')), np.array(f.get('/valid_data_id'))
with h5py.File(bad_data, 'r') as f:
    label_bad = np.array(f.get('/label'))


label_good, label_bad = make_iel_label(label_good, outCh), make_iel_label(label_bad, outCh)
out_train = np.concatenate((np.ones((train_data_id.shape[0], 1)), -1 * np.ones((train_data_id.shape[0], 1))), axis=0)
out_valid = np.concatenate((np.ones((valid_data_id.shape[0], 1)), -1 * np.ones((valid_data_id.shape[0], 1))), axis=0)
im_train = np.tile(im[train_data_id, ...], (2, 1, 1, 1))
im_valid = np.tile(im[valid_data_id, ...], (2, 1, 1, 1))
label_train = np.concatenate((label_good[train_data_id, ...], label_bad[train_data_id, ...]), axis=0)
label_valid = np.concatenate((label_good[valid_data_id, ...], label_bad[valid_data_id, ...]), axis=0)
print('Data is loaded. Training: %d, validation: %d' % (label_train.shape[0], label_valid.shape[0]))
im_train, label_train = np.tile(im_train, (10, 1, 1, 1)), np.tile(label_train, (10, 1, 1, 1))
out_train = np.tile(out_train, (10, 1))


gradients_i = tf.concat(tf.gradients(model_.outputs[0], model_.inputs), axis=-1)
gradients = tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(gradients_i), reduction_indices=[-1]), 1e-6, 1e6))


if numGPU > 1:
    model = multi_gpu_model(model_, gpus=numGPU)
else:
    model = model_
model.compile(loss=get(loss), optimizer=optimizers.rmsprop(lr=lr, clipnorm=1.0, clipvalue=1.0))  #


def load_batch_parallel(im, datasetID, nBatch, label=None, out=None, isAug=False, coord_sys='carts', isCritique=False,
                        prob_lim=0.5):
    n = len(datasetID)
    j = np.mod(np.arange(nBatch), n)
    while True:
        im_ = im[datasetID[j], ...].copy()
        if label is not None:
            label_ = label[datasetID[j], ...].copy()
        else:
            label_ = None
        if isAug:
            pool = ThreadPool(processes=cpu_count())
            multiple_results = [pool.apply_async(img_aug, (im_[[i], ...], label_[[i], ...], coord_sys, prob_lim))
                                for i in range(nBatch)]
            for i, res in enumerate(multiple_results):
                im_[i, ...], label_[i, ...] = res.get()
            pool.close()
        if not isCritique:
            yield (im_, label_)
        else:
            yield ([im_, label_], out[datasetID[j], :])
        j = np.mod(j + nBatch, n)


start = time.time()
for iEpoch in range(lastEpoch, nEpoch):
    print('Epoch %d / %d' % (iEpoch + 1, nEpoch))
    train_data_gen = load_batch_parallel(im_train, np.arange(im_train.shape[0]), nBatch, label_train, out_train,
                                         isAug=True, coord_sys='polar', prob_lim=(1 - iEpoch / nEpoch), isCritique=True)
    model.fit_generator(train_data_gen, steps_per_epoch=np.ceil(im_train.shape[0] / nBatch), verbose=1)
    if iEpoch % 10 == 9:
        loss_t = model.evaluate(x=[im_train, label_train], y=out_train, verbose=0, batch_size=nBatch)
        loss_v = model.evaluate(x=[im_valid, label_valid], y=out_valid, verbose=0, batch_size=nBatch)
        u0t = model.predict(x=[im_train, label_train], batch_size=nBatch)
        u0v = model.predict(x=[im_valid, label_valid], batch_size=nBatch)
        j = 0
        acc_t = (np.mean(u0t[out_train == 1] >= j) + np.mean(u0t[out_train == -1] < j)) / 2
        j = 0
        acc_v = (np.mean(u0v[out_valid == 1] >= j) + np.mean(u0v[out_valid == -1] < j)) / 2
        with open(log_file, 'a') as f:
            txt = "%d, %.2f, %f, %f, %f, %f \n" % (iEpoch + 1, (time.time() - start) / 3600.0, loss_t, loss_v, acc_t, acc_v)
            f.write(txt)
            print(txt)

    if iEpoch % 100 == 99:
        model_.save('model/' + exp_def + '/model-epoch%06d.h5' % (iEpoch + 1))

