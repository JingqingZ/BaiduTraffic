#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

event_traffic_file = "H:/event_traffic_beijing.pkl"
event_traffic_flag_file = "H:/event_traffic_flag_beijing.pkl"
RETAIN_RATIO_THRESHOLD = 0.9

print "loading event traffic ..."
event_traffic = cPickle.load(open(event_traffic_file, "rb"))
retain_rate_dict = defaultdict()
event_traffic_flag = defaultdict()
print "computing loss rate ..."
for link_id in event_traffic.keys():
    temp_traffic = event_traffic[link_id]
    temp_cnt = 0.0
    temp_len = len(temp_traffic)
    for time in range(temp_len):
        if temp_traffic[time] > 0:
            temp_cnt += 1
    retain_rate_dict[link_id] = temp_cnt / temp_len
    if temp_cnt/temp_len > RETAIN_RATIO_THRESHOLD:
        event_traffic_flag[link_id] = True
    else:
        event_traffic_flag[link_id] = False

cPickle.dump(event_traffic_flag, open(event_traffic_flag_file, "wb"))
total_len = len(retain_rate_dict)
bar_len = 10
bar_cnt = np.zeros(bar_len+1, np.float)
for value in retain_rate_dict.values():
    temp_index = int(value * 10)
    bar_cnt[temp_index] += 1

for i in range(bar_len+1):
    print "{}:\t{}".format(i, bar_cnt[i])

