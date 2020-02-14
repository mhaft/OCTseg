# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import os
import time
import csv
import glob
import argparse

import tifffile
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.losses import get

from unet.unet import unet_model
from unet.loss import weighted_cross_entropy_with_boundary_fun
from util.load_data import load_train_data
from util.load_batch import LoadBatchGenGPU
from util.read_parameter_from_log_file import read_parameter_from_log_file


def main():
    """Train or test a U-Net model to analyze OCT images.

    Notes:
        **All arguments are bash arguments**.

    Args:
        exp_def: experiment definition
        models_path: path for saving models
        lr: learning rate
        lr_decay: learning rate step for decay
        data_pat: data folder path
        nEpoch: number of epochs
        nBatch: batch size
        outCh: size of output channel
        inCh: size of input channel
        nZ: size of input depth
        w: size of input width (number of columns)
        l: size of input Length (number of rows)
        loss_w: loss wights
        isAug: Is data augmentation
        isCarts: whether images should be converted into Cartesian
        isTest: Is test run instead of train
        testEpoch: epoch of the saved model for testing
        saveEpoch: epoch interval to save the model
        epochSize: number of samples per epoch
        nFeature: number of features in the first layer
        nLayer: number of layers in the U-Nnet model
        gpu_id: ID of GPUs to be used
        optimizer: keras optimizer. see :meth:`keras.optimizers`

        See Also:
            * :meth:`unet.unet.unet_model`
            * :meth:`unet.loss.multi_loss_fun`
            * :meth:`util.load_data.load_train_data`
            * :meth:`util.load_batch.load_batch_parallel`
            * :meth:`keras.utils.multi_gpu_model`
            * :meth:`keras.optimizers`

    """

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_def", type=str, default="test", help="experiment definition")
    parser.add_argument("-models_path", type=str, default="model/", help="path for saving models")
    parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("-lr_decay", type=float, default=0.0, help="learning rate decay")
    parser.add_argument("-data_path", type=str, default="D:\\MLIntravascularPolarimetry\\MLCardioPullbacks\\",
                        help="data folder path")
    parser.add_argument("-nEpoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("-nBatch", type=int, default=10, help="batch size")
    parser.add_argument("-outCh", type=int, default=4, help="size of output channel")
    parser.add_argument("-inCh", type=int, default=3, help="size of input channel")
    parser.add_argument("-nZ", type=int, default=1, help="size of input depth")
    parser.add_argument("-w", type=int, default=512, help="size of input width (# of columns)")
    parser.add_argument("-l", type=int, default=512, help="size of input Length (# of rows)")
    parser.add_argument("-loss_w", type=str, default=".33, .33, .34", help="loss wights")
    parser.add_argument("-isAug", type=int, default=1, help="Is data augmentation")
    parser.add_argument("-isCarts", type=int, default=0, help="whether images should be converted into Cartesian")
    parser.add_argument("-isTest", type=int, default=0, help="Is test run instead of train. 1 when paramters are "
                                                             "from arguments. 2 when paramters are from log file.")
    parser.add_argument("-testEpoch", type=int, default=10, help="epoch of the saved model for testing")
    parser.add_argument("-saveEpoch", type=int, default=100, help="epoch interval to save the model")
    parser.add_argument("-epochSize", type=int, default=10, help="number of samples per epoch as multiple of the "
                                                                 "training dataset size")
    parser.add_argument("-nFeature", type=int, default=64, help="number of features in the first layer")
    parser.add_argument("-nLayer", type=int, default=8, help="number of layers in the U-Net model")
    parser.add_argument("-gpu_id", type=str, default="*", help="ID of GPUs to be used. Use * for all and '' for none.")
    parser.add_argument("-optimizer", type=str, default="RMSprop", help="optimizer")

    # assign the first part of args. The second part will ba assigned after reading parameter from log file
    args = parser.parse_args()
    experiment_def = args.exp_def
    isTest = args.isTest
    isTrain = 0 if args.isTest else 1
    models_path = args.models_path

    # prepare a folder for the saved models and log file
    if not os.path.exists(models_path + experiment_def):
        os.makedirs(models_path + experiment_def)
    save_file_name = models_path + experiment_def + '/model-epoch%06d.h5'
    log_file = models_path + experiment_def + '/log-' + experiment_def + '.csv'

    # read parameter from log file
    if args.isTest == 2:
        args = read_parameter_from_log_file(args, log_file)

    # assign the second part of args
    folder_path = args.data_path
    nEpoch = args.nEpoch
    nBatch = args.nBatch
    im_shape = (args.nZ, args.l, args.w, args.inCh)
    outCh = args.outCh
    loss_weight = np.array([float(i) for i in args.loss_w.split(',')], dtype='float32')
    loss_weight = loss_weight / np.sum(loss_weight)
    coord_sys = 'carts' if args.isCarts else 'polar'

    # initialize the log file or update the parameters
    if isTrain:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                L = f.readlines()
            L[0] = 'epoch, Time (hr), Train_Loss, Valid_Loss, ' + str(args) + '\n'
            with open(log_file, 'w') as f:
                f.writelines(L)
        else:
            with open(log_file, 'w') as f:
                f.write('epoch, Time (hr), Train_Loss, Valid_Loss, ' + str(args) + '\n')

    # GPU settings
    if '-' in args.gpu_id:
        numGPU = args.gpu_id.split('-')
        numGPU = int(numGPU[1]) - int(numGPU[0]) + 1
    elif '*' in args.gpu_id:
        numGPU = len(os.popen('nvidia-smi').read().split('+\n')) - 5
        args.gpu_id = ','.join([str(i) for i in range(numGPU)])
    else:
        numGPU = len(args.gpu_id.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # build the model
    model_template = unet_model(im_shape, nFeature=args.nFeature, outCh=outCh, nLayer=args.nLayer)

    if isTrain:
        # load the last saved model if exists
        f = glob.glob(models_path + experiment_def + '/model-epoch*.h5')
        f.sort()
        if len(f):
            iEpochStart = int(f[-1][-9:-3])
            model_template.load_weights(save_file_name % iEpochStart)
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=',', skipinitialspace=True)
                for row in reader:
                    if int(row['epoch']) == iEpochStart:
                        last_time = float(row['Time (hr)']) * 3600
            print('model at epoch %d is loaded.' % iEpochStart)
            iEpochStart += 1
            iEpoch = nEpoch  # in case training is done
        else:
            iEpochStart = 1
            last_time = 0
            print('Model is initialized.')
    elif isTest:
        iEpoch = args.testEpoch
        model_template.load_weights(save_file_name % iEpoch)
        print('model at epoch %d is loaded.' % args.testEpoch)

    if numGPU > 1:
        model = multi_gpu_model(model_template, gpus=numGPU)
    else:
        model = model_template

    optimizer = getattr(optimizers, args.optimizer)
    model.compile(optimizer=optimizer(lr=args.lr, decay=args.lr_decay),
                  loss=get(weighted_cross_entropy_with_boundary_fun(loss_weight)))


    # load data
    data_file = os.path.join(folder_path, 'Dataset ' + coord_sys + ' Z%d-L%d-W%d-C%d.h5' % im_shape)
    if os.path.exists(data_file):
        with h5py.File(data_file, 'r') as f:
            im, label, train_data_id, test_data_id, valid_data_id, sample_caseID = np.array(f.get('/im')), \
                np.array(f.get('/label')), np.array(f.get('/train_data_id')),  np.array(f.get('/test_data_id')), \
                np.array(f.get('/valid_data_id')), np.array(f.get('/sample_caseID'))
    else:
        im, label, train_data_id, test_data_id, valid_data_id, sample_caseID = \
            load_train_data(folder_path, im_shape, coord_sys)
        with h5py.File(data_file, 'w') as f:
            f.create_dataset('im', data=im)
            f.create_dataset('label', data=label)
            f.create_dataset('train_data_id', data=train_data_id)
            f.create_dataset('test_data_id', data=test_data_id)
            f.create_dataset('valid_data_id', data=valid_data_id)
            f.create_dataset('sample_caseID', data=sample_caseID)

    # labels and masks
    # Todo: add an input method for classes and masks
    # Intima layer
    loss_mask_classes = [0, 1]
    classes = [
        [[0, 1, 2, 4], []],
        [[3], [0]],
    ]

    # # Multi class
    # loss_mask_classes = []
    # classes = [
    #     [[1], [0, 2]],
    #     [[0], []],
    #     [[2], [0]],
    #     [[3], [0, 1]],
    # ]

    # # media
    # loss_mask_classes = []
    # classes = [
    #     [[0, 1, 2, 3], []],
    #     [[4], [0, 1]],
    # ]

    label_9class = label
    if loss_mask_classes:
        loss_mask = np.any(label_9class[..., loss_mask_classes], -1)
    else:
        loss_mask = np.zeros(label_9class.shape[:-1], dtype=np.bool)
    label = np.zeros(label.shape[:-1] + (len(classes),))
    label[..., 0] = np.all(np.logical_not(label_9class), axis=-1)
    for i in range(len(classes)):
        tmp = label[..., i]
        for j in range(len(classes[i][0])):
            tmp = np.logical_or(tmp, label_9class[..., classes[i][0][j]])
        for j in range(len(classes[i][1])):
            tmp = np.logical_and(tmp, np.logical_not(label_9class[..., classes[i][1][j]]))
        tmp = tmp.astype('float32')
        tmp[loss_mask] = 2
        label[..., i] = tmp

    # training
    if isTrain:
        train_data_gen = LoadBatchGenGPU(im, train_data_id, nBatch, label, isAug=args.isAug, coord_sys=coord_sys,
                                      prob_lim=0.5)
        if args.epochSize == 0:
            args.epochSize = np.ceil(train_data_id.size / nBatch).astype('int')
        else:
            args.epochSize = np.ceil(args.epochSize * train_data_id.size / nBatch).astype('int')
        print('Data is loaded. Training: %d, validation: %d' % (len(np.unique(sample_caseID[train_data_id])),
                                                                len(np.unique(sample_caseID[valid_data_id]))))

        start = time.time() - last_time
        for iEpoch in range(iEpochStart, nEpoch + 1):
            model.fit_generator(train_data_gen, steps_per_epoch=args.epochSize, verbose=1)
            # evaluation
            train_loss = model.evaluate(im[train_data_id, ...], label[train_data_id, ...],
                                        batch_size=nBatch, verbose=0)
            valid_loss = model.evaluate(im[valid_data_id, ...], label[valid_data_id, ...],
                                        batch_size=nBatch, verbose=0)
            rem_time = (nEpoch - iEpoch) / iEpoch * (time.time() - start) / 3600.0
            print("Epoch%d: %.2f hr to finish, Train Loss: %f, Valid Loss: %f" % (iEpoch, rem_time,
                                                                                  train_loss, valid_loss))
            with open(log_file, 'a') as f:
                f.write("%d, %.2f, %f, %f, \n" % (iEpoch, (time.time() - start) / 3600.0, train_loss, valid_loss))

            # save model
            if iEpoch % args.saveEpoch == 0:
                model_template.save(save_file_name % iEpoch)

    # feed forward
    train_valid_data_id = np.union1d(train_data_id, valid_data_id)
    out = model.predict(im, batch_size=nBatch, verbose=1)

    # see the loss for the first 20 slices
    LOSS = np.zeros((20, ) + label.shape[1:-1], dtype='float32')
    for i in tqdm(range(LOSS.shape[0])):
        LOSS[[i], ...] = K.eval(model.loss(tf.constant(label[[i], ...].astype('float32')),
                                           tf.constant((out[[i], ...]).astype('float32'))))

    out = np.argmax(out, -1)
    label = np.argmax(label, -1)
    if len(out.shape) > 3:
        i = int(out.shape[1] // 2)
        label, out, im = label[:, i, ...].squeeze(), out[:, i, ...].squeeze(), im[:, i, ...].squeeze()

    # set the label intensity of the training slices background to the number of classes, which is one more than
    # the last class value
    i = label[train_data_id, ...]
    i[i == 0] = outCh
    label[train_data_id, ...] = i

    # write files
    tifffile.imwrite(models_path + experiment_def + '/a-label.tif', label[train_valid_data_id, ...].astype(np.uint8))
    tifffile.imwrite(models_path + experiment_def + '/a-out-epoch%06d.tif' % iEpoch,
                     out[train_valid_data_id, ...].astype(np.uint8))
    tifffile.imwrite(models_path + experiment_def + '/a-im.tif',
                     (im[train_valid_data_id, ...] * 255).astype(np.uint8).squeeze())
    tifffile.imwrite(models_path + experiment_def + '/a-loss.tif', LOSS[train_valid_data_id, ...].astype('float32'))


if __name__ == '__main__':
    main()
