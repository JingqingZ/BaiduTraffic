
import os

def make_dirlist(dirlist):
    for dir in dirlist:
        if not os.path.exists(dir):
            os.makedirs(dir)

if __name__ == "__main__":
    pass
