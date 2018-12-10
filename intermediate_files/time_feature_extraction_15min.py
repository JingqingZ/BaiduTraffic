#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

"""
    time feature extraction
"""

root_dir = "../../Baidu/pkl/"
TIME_FEATURE_DIM = 6
output_time_feature_file = root_dir + "time_feature_15min.pkl"

"""
Monday - Sunday
day = [0,  1,
           2,  3,  4,  5,  6,  7,  8,
           9,  10,  11,  12,  13,  14,  15,
           16,  17,  18,  19,  20,  21,  22,
           23,  24,  25,  26,  27,  28,  29,
           30,  31,  32,  33,  34,  35,  36,
           37,  38,  39,  40,  41,  42,  43,
           44,  45,  46,  47,  48,  49,  50,
           51,  52,  53,  54,  55,  56,  57,
           58,  59,  60
]
"""
workday = [0,
           4,  5,  6,
           9,  10,  11,  12,  13,
           16,  17,  18,  19,  20,
           23,  24,  25,  26,  27,
           31,  32,  33,  34,
           37,  38,  39,  40,  41,
           44,  45,  46,  47,  48,
           51,  52,  53,  54,  55,  56,
           60
]
weekend = [
    7, 8,
    14, 15,
    21, 22,
    35, 36,
    42, 43,
    49, 50,
]
festival = [
    1, 2, 3,  # Tomb Sweeping Day
    28, 29, 30,  # May Day
    57, 58, 59  # Dragon Boat Festival
]
"""
    ### extract time feature ###
    period: 1 Apr, 2017 - 31 May, 2017
    workday, holiday(weekend or festival): one-hot, 3 dim
    hour: float, 1 dim
    min: float, 1 dim
    peak hour(7:00-10:00, 17:00-20:00): float, 1 dim
    time_feature_dim = 6
"""


def time_feature_extraction(time):
    temp_time_feature = np.zeros(TIME_FEATURE_DIM, dtype=np.float)
    day = int(time / (24 * 4))
    if day in workday:
        temp_time_feature[0] = 1
    elif day in weekend:
        temp_time_feature[1] = 1
    else:
        temp_time_feature[2] = 1
    hour = int((time - day * 24 * 4) / 4)
    min = time - day * 24 * 4 - hour * 4
    temp_time_feature[3] = hour
    temp_time_feature[4] = min
    if (hour in range(6, 9)) or (hour in range(16, 19)):
        temp_time_feature[5] = 1
    return temp_time_feature


TOTAL_TIME = 61 * 24 * 4  # 61 days * 24 hours * 4 mins(15 min interval)
time_feature = np.zeros((TOTAL_TIME, TIME_FEATURE_DIM), dtype=np.float)
for time in range(TOTAL_TIME):
    time_feature[time, :] = time_feature_extraction(time)

cPickle.dump(time_feature, open(output_time_feature_file, "wb"))
