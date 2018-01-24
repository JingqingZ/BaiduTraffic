
import os
from datetime import datetime

def make_dirlist(dirlist):
    for dir in dirlist:
        if not os.path.exists(dir):
            os.makedirs(dir)

time_fmt = "%Y-%m-%d-%H-%M-%S"

def now2string(fmt="%Y-%m-%d-%H-%M-%S"):
    return datetime.now().strftime(fmt)

if __name__ == "__main__":
    pass
