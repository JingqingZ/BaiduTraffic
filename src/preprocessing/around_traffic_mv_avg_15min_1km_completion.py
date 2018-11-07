#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

root_dir = "../../Baidu/pkl/"
event_traffic_file = root_dir + "event_traffic_beijing_1km_mv_avg_15min.pkl"
event_traffic_completion_file = root_dir + "event_traffic_beijing_1km_mv_avg_15min_completion.pkl"

print("loading event traffic ...")
event_traffic = cPickle.load(open(event_traffic_file, "rb"))

print("computing global average week traffic ...")
total_len = 61 * 24 * 4
week_len = 7 * 24 * 4
global_week_traffic = np.zeros(week_len, np.float)
global_week_cnt = np.zeros(week_len, np.float)
for link_id in event_traffic.keys():
    temp_traffic = event_traffic[link_id]
    temp_len = len(temp_traffic)
    for time in range(temp_len):
        if temp_traffic[time] > 0:
            week_time = time % week_len
            global_week_traffic[week_time] += temp_traffic[time]
            global_week_cnt[week_time] += 1
for time in range(week_len):
    global_week_traffic[time] = global_week_traffic[time]/global_week_cnt[time]

print("completing event link traffic ...")
event_traffic_completion = defaultdict()
cnt = 0
for link_id in event_traffic.keys():
    cnt += 1
    print("cnt is {}".format(cnt))
    temp_traffic = event_traffic[link_id]
    temp_len = len(temp_traffic)
    temp_week_traffic = np.zeros(week_len, np.float)
    temp_week_cnt = np.zeros(week_len, np.float)
    for time in range(temp_len):
        if temp_traffic[time] > 0:
            week_time = time % week_len
            temp_week_traffic[week_time] += temp_traffic[time]
            temp_week_cnt[week_time] += 1
    for time in range(week_len):
        if temp_week_cnt[time] > 0:
            temp_week_traffic[time] /= temp_week_cnt[time]
    # completion
    for time in range(temp_len):
        if temp_traffic[time] < 1:
            week_time = time % week_len
            if temp_week_traffic[week_time] > 0:
                temp_traffic[time] = temp_week_traffic[week_time]
            else:
                temp_traffic[time] = global_week_traffic[week_time]
    event_traffic_completion[link_id] = temp_traffic

cPickle.dump(event_traffic_completion, open(event_traffic_completion_file, "wb"))
