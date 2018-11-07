#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np
import time

root_dir = "F:/"
input_dir = "H:/traffic_beijing_road_1km_10000"
event_link_set_input_file = root_dir + "event_link_set_beijing_1km_10000"
event_traffic_file = root_dir + "event_traffic_beijing_1km_mv_avg_15min.pkl"

all_link_set = set()
for line in open(event_link_set_input_file):
    temp_arr = line.strip("\n").split("\t")
    for temp_link in temp_arr:
        all_link_set.add(temp_link)
len_all_link = len(all_link_set)
print("len(all_link_set) is {}".format(len_all_link))

min_num = 60 * 24 * 61
moving_average_window = 30
average_window = 15
event_traffic = defaultdict()
cnt = 0
for temp_link in all_link_set:
    complete_cnt = 0
    cnt += 1
    temp_list = np.zeros(min_num, np.float)
    cnt_list = np.zeros(min_num, np.float)
    temp_list_15_min = np.zeros(min_num / average_window, np.float)
    try:
        start = time.clock()
        for line in open(os.path.join(input_dir, temp_link)):
            temp_arr = line.split(",")
            mon = int(temp_arr[6][4:6])
            day = int(temp_arr[6][6:8])
            hour = int(temp_arr[6][8:10])
            mins = int(temp_arr[6][10:12])
            time_id = 60 * (24 * (30 * (mon - 4) + (day - 1)) + hour) + mins
            if time_id >= 87840:
                print("%d\t%d\t%d\t%d" % (mon, day, hour, mins))
                print("time_id is %d" % time_id)
            if 150 > float(temp_arr[4]) > 0:
                temp_list[time_id] += float(temp_arr[4])
                cnt_list[time_id] += 1
        end1 = time.clock()

        for i in range(min_num):
            if cnt_list[i] > 0:
                temp_list[i] = temp_list[i] / cnt_list[i]
                cnt_list[i] = 1

        # moving average
        temp_sum = np.convolve(temp_list, np.ones((moving_average_window, )), mode='same')
        temp_sum_cnt = np.convolve(cnt_list, np.ones((moving_average_window, )), mode='same')
        temp_list_mv_avg = temp_sum / np.maximum(temp_sum_cnt, 1)
        # print temp_list_mv_avg[:10]
        end2 = time.clock()

        # average
        for i in range(min_num/average_window):
            temp_speed = 0.0
            temp_cnt = 0.0
            for j in range(average_window):
                temp_speed += temp_list_mv_avg[i*average_window + j]
                if temp_list_mv_avg[i*average_window + j] > 0:
                    temp_cnt += 1
            if temp_cnt > 0:
                complete_cnt += 1
                temp_speed = temp_speed / temp_cnt
            temp_list_15_min[i] = temp_speed
        if complete_cnt > min_num / average_window * 0.9:
            event_traffic[temp_link] = temp_list_15_min
            print("processing {}/{}, link id is {}, complete_cnt is {}".format(cnt, len_all_link, temp_link,
                                                                               complete_cnt))
        end3 = time.clock()
        # print("cost time: reading is {}, mv_avg is {}, avg is {}".format(end1-start, end2-end1, end3-end2))
    except:
        continue
print("valid_link_num is {}".format(len(event_traffic.keys())))
cPickle.dump(event_traffic, open(event_traffic_file, "wb"))
