# Copyright (C) 2019 Harvard University. All Rights Reserved. Unauthorized
# copying of this file, via any medium is strictly prohibited Proprietary and
# confidential
# Developed by Mohammad Haft-Javaherian <mhaft-javaherian@mgh.harvard.edu>,
#                                       <7javaherian@gmail.com>.
# ==============================================================================

"""Read parameters from the header line of the log file"""

import re
import warnings


def read_parameter_from_log_file(args, log_file):
    """
        Read parameters from the header line of the log file
    Args:
        args: inital args
        log_file: the log file address

    Returns:
        return the updated arg

    """

    f = open(log_file, 'r')
    params = re.findall(r"\b[\w-]+\=\'.*?\'[,)]|\b[\w-]+\=.*?[,)]", f.readline())
    f.close()
    testEpoch = args.testEpoch  # it should not get updated.

    for param in params:
        a, v = param.split("=")
        v = v.strip('\'",)')
        if v.isnumeric() or (v[0] == '-' and v[1:].isnumeric()):
            v = int(v)
        elif ''.join(re.split(r'[.eE\-]', v)).isnumeric():
            v = float(v)
        if hasattr(args, a):
            setattr(args, a, v)
        else:
            warnings.warn('The parameter %s in the log file is deprecated. log file: %s ' % (a, log_file))

    args.testEpoch = testEpoch
    return args
