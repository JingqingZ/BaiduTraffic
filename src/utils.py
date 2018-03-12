
import os
import numpy as np
from datetime import datetime

def make_dirlist(dirlist):
    for dir in dirlist:
        if not os.path.exists(dir):
            os.makedirs(dir)

time_fmt = "%Y-%m-%d-%H-%M-%S"

def now2string(fmt="%Y-%m-%d-%H-%M-%S"):
    return datetime.now().strftime(fmt)

def mape(pred, target):
    return np.abs(pred - target) / target

if __name__ == "__main__":
    pass
