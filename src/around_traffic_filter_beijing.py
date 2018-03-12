#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

root_dir = "../../data/"
input_event_beijing_file = root_dir + "event_beijing_extended_filtered.txt"
event_link_list_file = root_dir+"event_link_list_beijing.pkl"
event_traffic_flag_file = root_dir+"event_traffic_flag_beijing.pkl"
event_traffic_detrended_normalized_file = root_dir+"event_traffic_completion_beijing.pkl"

event_reshaped_traffic_list_file = root_dir+"event_filtered_traffic_list_beijing_nofilt.pkl"
event_average_traffic_list_file = root_dir+"event_average_traffic_list_beijing_nofilt.pkl"
output_event_beijing_file = root_dir + "event_beijing_final_2.txt"
output_event_top5_link_list_beijing_file = root_dir + "event_top5_link_list.pkl"
EVENT_NUM = 1151

print "loading event link list ..."
event_link_list = cPickle.load(open(event_link_list_file, "rb"))
print "event_link_list len is {}".format(len(event_link_list))

print "loading input event beijing ..."
input_event_beijing = list()
for line in open(input_event_beijing_file):
    input_event_beijing.append(line)
print "input_event_beijing len is {}".format(len(input_event_beijing))

print "loading event traffic ..."
event_traffic_detrend = cPickle.load(open(event_traffic_detrended_normalized_file, "rb"))
event_traffic_flag = cPickle.load(open(event_traffic_flag_file, "rb"))

print "preparing data ..."
valid_event_cnt = 0
# fw = open(output_event_beijing_file, "wb")
event_top5_link_list = list()
total_samples = 61 * 24 * 12
filter_num = 5
event_reshaped_traffic_list = list()
event_average_traffic_list = list()
filterlist = list()
for event_id in range(EVENT_NUM):

    temp_link_set = event_link_list[event_id]
    temp_valid_link_num = 0
    for temp_link in temp_link_set:
        if temp_link in event_traffic_flag and event_traffic_flag[temp_link]:
            temp_valid_link_num += 1

    if temp_valid_link_num < 5:
        print "event_id is {},\tlink_num is {} ".format(event_id, temp_valid_link_num)
        filterlist.append(0)
        continue
    temp_event_total_x = np.zeros((total_samples, temp_valid_link_num), dtype=np.float)
    # fw.write(input_event_beijing[event_id])
    valid_event_cnt += 1
    temp_cnt = 0
    link_dict = defaultdict()
    for temp_link in temp_link_set:
        if temp_link in event_traffic_flag and event_traffic_flag[temp_link]:
            temp_link_traffic = event_traffic_detrend[temp_link]
            link_dict[temp_link] = np.sum(temp_link_traffic)
            temp_event_total_x[:, temp_cnt] = temp_link_traffic[:]
            temp_cnt += 1

    # select the top 5 heavy link
    sorted_list = sorted(link_dict.items(), lambda x, y: cmp(x[1], y[1]))
    filtered_event_traffic = np.zeros((total_samples, filter_num), dtype=np.float)
    temp_link_top5_set = set()
    for i in range(filter_num):
        print " link id is {} ".format(sorted_list[i][0]),
        temp_link_top5_set.add(sorted_list[i][0])
        filtered_event_traffic[:, i] = event_traffic_detrend[sorted_list[i][0]]
    print ""
    event_top5_link_list.append(temp_link_top5_set)
    filterlist.append(1)
    # detrend the traffic
    event_reshaped_traffic_list.append(filtered_event_traffic)
    event_mean_traffic = np.mean(temp_event_total_x, axis=1)
    event_average_traffic_list.append(event_mean_traffic)


print "valid_event_cnt/EVENT_NUM:{}/{}".format(valid_event_cnt, EVENT_NUM)
print "dumping event_traffic_list ..."
filterfile = open(root_dir + "event_filter.txt", "w")
filterfile.write(str(filterlist))
filterfile.close()
# fw.close()
# cPickle.dump(event_reshaped_traffic_list, open(event_reshaped_traffic_list_file, "wb"))
# cPickle.dump(event_average_traffic_list, open(event_average_traffic_list_file, "wb"))
# cPickle.dump(event_top5_link_list, open(output_event_top5_link_list_beijing_file, "wb"))
