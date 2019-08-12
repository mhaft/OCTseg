# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""plot the log file within the save model folders."""

import argparse

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument("-exp_def", type=str, default="test", help="experiment definition")
args = parser.parse_args()
log_file = '../model/' + args.exp_def + '/log-' + args.exp_def + '.csv'


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    try:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',', skipinitialspace=True)
            data = []
            next(reader)
            for row in reader:
                data.append([int(row['epoch']), float(row['Time (hr)']), float(row['Test_Loss']), float(row['Valid_Loss'])])
        ax1.clear()
        data = np.array(data)
        ax1.plot(data[-100:, 0], data[-100:, 2])
        ax1.plot(data[-100:, 0], data[-100:, 3])
        ax1.legend(['Test_Loss', 'Valid_Loss'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
    finally:
        pass


ani = animation.FuncAnimation(fig, animate, interval=5000)
plt.show()

