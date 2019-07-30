# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft_javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Convert an 2D or 3D image from polar or cylindrical coordinate to the
    cartesian coordinate."""

from __future__ import absolute_import, division, print_function

import os
import time

import tifffile
import argparse
import h5py
import numpy as np
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.losses import get
from unet.unet import unet_model
from unet.ops import load_batch
from unet.loss import multi_loss_fun
from util.load_data import load_train_data

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-exp_def", type=str, default="test", help="experiment definition")
parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-lr_decay", type=float, default=0.0, help="learning rate step for decay")
parser.add_argument("-data_path", type=str, default="C:\\MachineLearning\\PSTIFS\\", help="data folder path")
parser.add_argument("-nEpoch", type=int, default=2000, help="number of epochs")
parser.add_argument("-nBatch", type=int, default=5, help="batch size")
parser.add_argument("-outCh", type=int, default=2, help="size of output channel")
parser.add_argument("-inCh", type=int, default=1, help="size of input channel")
parser.add_argument("-nZ", type=int, default=1, help="size of input depth")
parser.add_argument("-w", type=int, default=512, help="size of input width (# of columns)")
parser.add_argument("-l", type=int, default=512, help="size of input Length (# of rows)")
parser.add_argument("-loss_w", type=str, default="1, 100", help="loss wights")
parser.add_argument("-isAug", type=int, default=1, help="Is data augmentation")
parser.add_argument("-isCarts", type=int, default=0, help="whether images should be converted into Cartesian")
parser.add_argument("-isTest", type=int, default=0, help="Is test run instead of train")
parser.add_argument("-testEpoch", type=int, default=0, help="epoch of the saved model for testing")
parser.add_argument("-saveEpoch", type=int, default=500, help="epoch interval to save the model")
parser.add_argument("-logEpoch", type=int, default=100, help="epoch interval to save the log")
parser.add_argument("-nFeature", type=int, default=32, help="number of features in the first layer")
parser.add_argument("-gpu_id", type=str, default="0,1", help="ID of GPUs to be used")

args = parser.parse_args()
experiment_def = args.exp_def
folder_path = args.data_path
nEpoch = args.nEpoch
nBatch = args.nBatch
im_shape = (args.nZ, args.l, args.w, args.inCh)
outCh = args.outCh
isTest = args.isTest
loss_weight = np.array([float(i) for i in args.loss_w.split(',')])
loss_weight = loss_weight / np.linalg.norm(loss_weight)
coord_sys = 'carts' if args.isCarts else 'polar'

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=config))
if '-' in args.gpu_id:
    numGPU = args.gpu_id.split('-')
    numGPU = int(numGPU[1]) - int(numGPU[0]) + 1
else:
    numGPU = len(args.gpu_id.split(','))

# prepare a folder for the saved models and log file
if not os.path.exists('model/' + experiment_def):
    os.makedirs('model/' + experiment_def)
save_file_name = 'model/' + experiment_def + '/model-epoch%06d.h5'
log_file = 'model/' + experiment_def + '/log-' + experiment_def + '.csv'
if not isTest:
    with open(log_file, 'w') as f:
        f.write('epoch, Time (hr), Test_Loss, Valid_Loss, ' + str(args) + '\n')

# build the model
model = unet_model(im_shape, nFeature=args.nFeature, outCh=outCh)
loss = multi_loss_fun(loss_weight)
if numGPU > 1:
    model = multi_gpu_model(model, gpus=numGPU)
if not isTest:
    save_callback = ModelCheckpoint(save_file_name)
    model.compile(optimizer=Adam(lr=args.lr), loss=get(loss))
    print('Model is initialized.')
else:
    model.load_weights(save_file_name % args.testEpoch)
    print('model at epoch %d is loaded.' % args.testEpoch)

# load data
data_file = os.path.join(folder_path, 'Dataset ' + coord_sys + ' Z%d-L%d-W%d-C%d.h5' % im_shape)
if os.path.exists(data_file):
    with h5py.File(data_file, 'r') as f:
        im, label, train_data_id, test_data_id, valid_data_id, sample_caseID = np.array(f.get('/im')), \
            np.array(f.get('/label')), np.array(f.get('/train_data_id')),  np.array(f.get('/test_data_id')), \
            np.array(f.get('/valid_data_id')), np.array(f.get('/sample_caseID'))
else:
    im, label, train_data_id, test_data_id, valid_data_id, sample_caseID = load_train_data(folder_path, im_shape,
                                                                                           coord_sys)
    with h5py.File(data_file, 'w') as f:
        f.create_dataset('im', data=im)
        f.create_dataset('label', data=label)
        f.create_dataset('train_data_id', data=train_data_id)
        f.create_dataset('test_data_id', data=test_data_id)
        f.create_dataset('valid_data_id', data=valid_data_id)
        f.create_dataset('sample_caseID', data=sample_caseID)
if not isTest:
    train_data_gen = load_batch(im, train_data_id, nBatch, label, isAug=True, coord_sys=coord_sys)
    valid_data_gen = load_batch(im, valid_data_id, nBatch, label, isAug=False, coord_sys=coord_sys)
    print('Data is loaded. Training: %d, validation: %d' % (len(np.unique(sample_caseID[train_data_id])),
                                                            len(np.unique(sample_caseID[valid_data_id]))))

# testing
if not isTest:
    start = time.time()
    for iEpoch in range(nEpoch):
        x1, l1 = next(train_data_gen)
        model.train_on_batch(x1, l1)
        if (iEpoch + 1) % args.logEpoch == 0:
            train_loss = model.evaluate(im[train_data_id, ...], label[train_data_id, ...], batch_size=nBatch, verbose=0)
            valid_loss = model.evaluate(im[valid_data_id, ...], label[valid_data_id, ...], batch_size=nBatch, verbose=0)
            rem_time = (nEpoch - iEpoch - 1) / (iEpoch + 1.0) * (time.time() - start) / 3600.0
            print("Epoch:%d, %.2f hr to finish, Train Loss: %f, Test Loss: %f" % (iEpoch + 1, rem_time,
                                                                                  train_loss, valid_loss))
            with open(log_file, 'a') as f:
                f.write("%d, %.2f, %f, %f, \n" % (iEpoch + 1, (time.time() - start) / 3600.0, train_loss, valid_loss))
        if (iEpoch + 1) % args.saveEpoch == 0:
            model.save(save_file_name % (iEpoch + 1))

# feed forward
label = np.argmax(label, -1)
train_valid_data_id = np.union1d(train_data_id, valid_data_id)
label[train_data_id, -15:, -10:-5] = 1
label[train_data_id, -20:-15, -15:] = 1
out = model.predict(im, batch_size=nBatch, verbose=1)
out = np.reshape(np.argmax(out, -1), im.shape[:-1])
tifffile.imwrite('model/' + experiment_def + '/a-label.tif', label[train_valid_data_id, ...].astype(np.uint8))
tifffile.imwrite('model/' + experiment_def + '/a-out.tif', out[train_valid_data_id, ...].astype(np.uint8))
tifffile.imwrite('model/' + experiment_def + '/a-im.tif', (im[train_valid_data_id, ...] * 255).astype(np.uint8).squeeze())


# if __name__ == '__main__':
#     pass
