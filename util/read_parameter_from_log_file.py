# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Read parameters from the header line of the log file"""


def read_parameter_from_log_file(args, log_file):
    """
        Read parameters from the header line of the log file
    Args:
        args: inital args
        log_file: the log file address

    Returns:
        return the updated arg

    """

    with open(log_file) as f:
        line = f.readline().split(',')
    for x in line:
        x = x.strip(' \t\n\r)')
        if x.startswith("Namespace"):
            x = x[10:]
        if x.startswith("data_path="):
            args.data_path = x[11:-1]
        elif x.startswith("l="):
            args.l = int(x[2:])
        elif x.startswith("w="):
            args.w = int(x[2:])
        elif x.startswith("inCh="):
            args.inCh = int(x[5:])
        elif x.startswith("nZ="):
            args.nZ = int(x[3:])
        elif x.startswith("outCh="):
            args.outCh = int(x[6:])
        elif x.startswith("isCarts="):
            args.isCarts = int(x[8:])
        elif x.startswith("epochSize="):
            args.epochSize = int(x[10:])
        elif x.startswith("nBatch="):
            args.nBatch = int(x[7:])
        elif x.startswith("nEpoch="):
            args.nEpoch = int(x[7:])
        elif x.startswith("nFeature="):
            args.nFeature = int(x[9:])
        elif x.startswith("nLayer="):
            args.nLayer = int(x[7:])
        elif x.startswith("saveEpoch="):
            args.saveEpoch = int(x[10:])
        elif x.startswith("critique_model="):
            args.critique_model = x[16:-1]
        elif x.startswith("critiqueEpoch="):
            args.critique_model = int(x[14:])

    with open(log_file) as f:
        line = f.readline().split('\'')
    for i in range(len(line)):
        if line[i].endswith("loss_w="):
            args.loss_w = line[i + 1]
            break
    return args