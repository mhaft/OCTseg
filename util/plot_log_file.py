# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""plot the log file within the save model folder. 111

    plots the train and validation loss values over last 100 recorded performance evaluations and update the
    plot every 5 second.  The figure has two subplots: top one has all the results and bottom one has last 100 log
    records.

    Notes:
        Arguments are bash arguments.

    Args:
        exp_def: the experiment definition used for saving the model.
        models_path: the path that model folder for `exp_def`

    Returns:
        PyPlot figure with two subplots.

    See Also:
        * :meth:`train`

    """

import argparse
import time
import os
import csv

import visdom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def smooth(x):
    """Smoothing 1D using box filter with kernel size = 5

    Args:
        x: input 1D vector

    Returns:
        smoothed 1D vector

    """
    s = np.concatenate((np.tile(x[0], 4), x))
    s[4:] = (s[4:] + s[3:-1] + s[2:-2] + s[1:-3] + s[:-4]) / 5
    return s[4:]


def animate(i):
    """ a handle function to update at each ste

    Args:
        i: animation frame.  The argument will be used by :meth: animation.FuncAnimation

    Returns:
        updated axes within the figure, which all are defined in the outer scope.

    See Also:
        * :meth: animation.FuncAnimation

    """
    global numRecord
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            _ = next(reader)  # header
            data = []
            for row in reader:
                data.append([float(i) for i in row[:4]])
        if len(data) != numRecord:
            numRecord = len(data)
            ax1.clear()
            ax2.clear()
            data = np.array(data)
            iStart = 0
            ax1.plot(data[iStart:, 0], smooth(data[iStart:, 2]))
            ax1.plot(data[iStart:, 0], smooth(data[iStart:, 3]))
            ax1.legend(['Training Loss', 'Validation Loss'])
            ax1.set_title(args.exp_def)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            iStart = -50 if data.shape[0] > 50 else 0
            ax2.plot(data[iStart:, 0], data[iStart:, 2] / data[iStart, 2])
            ax2.plot(data[iStart:, 0], data[iStart:, 3] / data[iStart, 3])
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
    except:
        pass


def animate_vis():
    """ a handle function to update at each ste

    Args:
        iEpoch: animation frame.  The argument will be used by :meth: animation.FuncAnimation

    Returns:
        updated axes within the figure, which all are defined in the outer scope.

    See Also:
        * :meth: animation.FuncAnimation

    """
    global numRecord
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            _ = next(reader)  # header
            data = []
            for row in reader:
                data.append([float(i) for i in row[:4]])

        if len(data) != numRecord:
            numRecord = len(data)
            data = np.array(data)
            iStart = 0
            vis.line(Y=np.concatenate((smooth(data[iStart:, 2])[..., np.newaxis],
                                      smooth(data[iStart:, 3])[..., np.newaxis]), axis=-1), X=data[iStart:, 0],
                     env=vis_env, win=args.exp_def, opts=dict(title=args.exp_def, xlabel='Epoch', ylabel='Loss',
                                                             legend=['Training Loss', 'Validation Loss']))
            iStart = -50 if data.shape[0] > 50 else 0
            vis.line(Y=np.concatenate(((data[iStart:, 2] / data[iStart, 2])[..., np.newaxis],
                                       (data[iStart:, 3] / data[iStart, 3])[..., np.newaxis]), axis=-1),
                     X=data[iStart:, 0],
                     env=vis_env, win=args.exp_def+'-zoom', opts=dict(title=args.exp_def+' zoom', xlabel='Epoch',
                                                                 ylabel='Loss',
                                                             legend=['Training Loss', 'Validation Loss']))
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_def", type=str, default="test", help="experiment definition")
    parser.add_argument("-models_path", type=str, default="../model/", help="path for saving models")
    parser.add_argument("-render", type=str, default="visdom", help="plot environment: visdom (v) or pyplot (p)")
    parser.add_argument("-nvidia_smi", type=int, default=0, help="print output of nvidia-smi")
    args = parser.parse_args()
    log_file = args.models_path + args.exp_def + '/log-' + args.exp_def + '.csv'
    nGPU = len(os.popen('nvidia-smi').read().split('+\n')) - 5
    numRecord = 0
    if args.render[0].lower() == 'v':
        vis_env = 'haft'
        vis = visdom.Visdom(env=vis_env)
        while True:
            animate_vis()
            if args.nvidia_smi:
                smi = '<p style="color:blue;font-family:monospace;font-size:80%;">' + \
                      '<br>'.join(os.popen('nvidia-smi').read().split('\n')[3:(7 + 3 * nGPU)]) + '</p>'
                vis.text(smi, env=vis_env, win='nvidia-smi')
            time.sleep(5)
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        animate(-1)
        ani = animation.FuncAnimation(fig, animate, interval=5000)
        plt.show()


