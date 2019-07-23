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
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from unet.unet import unet_model
from unet.ops import load_batch
from unet.loss import cross_entropy
from util.load_data import load_train_data

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-exp_def", type=str, default="test", help="experiment definition")
parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-lr_step", type=int, default=1000, help="learning rate step for decay")
parser.add_argument("-data_path", type=str, default="C:\\MachineLearning\\PSTIFS\\", help="data folder path")
parser.add_argument("-nEpoch", type=int, default=2000, help="number of epochs")
parser.add_argument("-nBatch", type=int, default=5, help="batch size")
parser.add_argument("-outCh", type=int, default=2, help="size of output channel")
parser.add_argument("-inCh", type=int, default=1, help="size of input channel")
parser.add_argument("-nZ", type=int, default=1, help="size of input depth")
parser.add_argument("-w", type=int, default=512, help="size of input width")
parser.add_argument("-loss_w", type=str, default="1, 100, 0", help="loss wights")
parser.add_argument("-isAug", type=int, default=1, help="Is data augmentation")
parser.add_argument("-saveEpoch", type=int, default=500, help="epoch interval to save the model")
parser.add_argument("-logEpoch", type=int, default=100, help="epoch interval to save the log")
parser.add_argument("-nFeature", type=int, default=32, help="number of features in the first layer")
parser.add_argument("-gpu_id", type=str, default="0,1", help="ID of GPUs to be used")

args = parser.parse_args()
experiment_def = args.exp_def
folder_path = args.data_path
nEpoch = args.nEpoch
nBatch = args.nBatch
im_shape = (args.nZ, args.w, args.w, args.inCh)
outCh = args.outCh
loss_weight = [float(i) for i in args.loss_w.split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if not os.path.exists('model/' + experiment_def):
    os.makedirs('model/' + experiment_def)
save_file_name = 'model/' + experiment_def + '/model-epoch%06d.h5'
log_file = 'model/' + experiment_def + '/log-' + experiment_def + '.csv'
with open(log_file, 'w') as f:
    f.write('epoch, Time (hr), Test_Loss, Valid_Loss, ' + str(args) + '\n')

model = unet_model(im_shape, nFeature=args.nFeature, outCh=outCh)

if '-' in args.gpu_id:
    numGPU = args.gpu_id.split('-')
    numGPU = int(numGPU[1]) - int(numGPU[0]) + 1
else:
    numGPU = len(args.gpu_id.split(','))
if numGPU > 1:
    model = multi_gpu_model(model, gpus=numGPU)
    nBatch *= numGPU

save_callback = ModelCheckpoint(save_file_name)
model.compile(optimizer=Adam(lr=args.lr), loss=cross_entropy)
print('Model is initialized.')

data_file = os.path.join(folder_path, 'dataset Z%d-W%d-L%d-C%d' % im_shape)
if os.path.exists(data_file):
    with h5py.File(data_file, 'r') as f:
        im, label, train_data_id, test_data_id, valid_data_id = np.array(f.get('/im')), np.array(f.get('/label')), \
            np.array(f.get('/train_data_id')),np.array(f.get('/test_data_id')), np.array(f.get('/valid_data_id'))
else:
    im, label, train_data_id, test_data_id, valid_data_id = load_train_data(folder_path, im_shape)
    with h5py.File(data_file, 'w') as f:
        f.create_dataset('im', data=im)
        f.create_dataset('label', data=label)
        f.create_dataset('train_data_id', data=train_data_id)
        f.create_dataset('test_data_id', data=test_data_id)
        f.create_dataset('valid_data_id', data=valid_data_id)
train_data_gen = load_batch(im, train_data_id, nBatch, label, isAug=True)
valid_data_gen = load_batch(im, valid_data_id, nBatch, label, isAug=False)
print('Data is loaded. Training: %d, validation: %d' % (len(train_data_id), len(valid_data_id)))

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
        model.save(save_file_name%(iEpoch + 1))

label = np.argmax(label, -1)
out = model.predict(im, batch_size=nBatch)
out = np.reshape(np.argmax(out, -1), im.shape)
tifffile.imwrite('model/' + experiment_def + '/a-label.tif', label.astype(np.uint8))
tifffile.imwrite('model/' + experiment_def + '/a-out.tif', out.astype(np.uint8))
tifffile.imwrite('model/' + experiment_def + '/a-im.tif', (im * 255).astype(np.uint8).squeeze())


# if __name__ == '__main__':
#     pass
