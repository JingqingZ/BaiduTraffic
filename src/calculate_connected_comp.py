#!/usr/bin/python
# coding=utf-8

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import progressbar
from collections import defaultdict

"""
    use union-find algorithm to calculate the num of connected components of the beijing road network
"""

# Binbing
# datapath = "F:/"
# Jingqing
datapath = "../../data/"
pre_dict = defaultdict()


def find(a):
    if(pre_dict[a] != a):
        pre_dict[a] = find(pre_dict[a])
    return pre_dict[a]


def join(x, y):
    fx = find(x)
    fy = find(y)
    if(fx!=fy):
        pre_dict[fy] = fx


def roadnet_extraction():
    roadnetfilename = datapath + "beijing roadnet/R.mid"
    # get the node
    for line in open(roadnetfilename):
        temp_arr = line.replace("\"", "").split("\t")
        if temp_arr[1] not in pre_dict:
            pre_dict[temp_arr[1]] = temp_arr[1]
        if temp_arr[9] not in pre_dict:
            pre_dict[temp_arr[9]] = temp_arr[9]
        if temp_arr[10] not in pre_dict:
            pre_dict[temp_arr[10]] = temp_arr[10]
    print("len(pred_dict) is {}".format(len(pre_dict)))

    for line in open(roadnetfilename):
        temp_arr = line.replace("\"", "").split("\t")
        join(temp_arr[9], temp_arr[1])
        join(temp_arr[1], temp_arr[10])

    num_connected_comp = 0
    for key in pre_dict:
        if pre_dict[key] == key:
            num_connected_comp += 1

    print("num_connected_comp is {}".format(num_connected_comp))


if __name__ == "__main__":
    roadnet_extraction()