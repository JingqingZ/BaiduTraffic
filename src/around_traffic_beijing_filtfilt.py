#!/usr/bin/python
# coding=utf-8
import cPickle
from scipy import signal
from collections import defaultdict

root_dir = "../../data/"
event_traffic_completion_file = root_dir + "event_traffic_completion_beijing_15min.pkl"
event_traffic_flag_file = root_dir + "event_traffic_flag_beijing.pkl"
event_traffic_completion_filtfilt_file = root_dir + "event_traffic_completion_beijing_15min_filtfilt_0.05.pkl"
print("loading event_traffic data ...")

event_traffic = cPickle.load(open(event_traffic_completion_file, "rb"))
event_traffic_flag = cPickle.load(open(event_traffic_flag_file, "rb"))
print("loading completed.")

event_traffic_completion_filtfilt = defaultdict()
# hyper-para
b, a = signal.butter(8, 0.05)
len_before = 0
len_after = 0
cnt = 0
for link_id in event_traffic.keys():
    if not event_traffic_flag[link_id]:
        continue
    cnt += 1
    print("valid link num is {}".format(cnt))
    temp_query_list = event_traffic[link_id]
    len_before = len(temp_query_list)
    new_temp_query_list = signal.filtfilt(b, a, temp_query_list, padlen=23)
    len_after = len(new_temp_query_list)
    event_traffic_completion_filtfilt[link_id] = new_temp_query_list

cPickle.dump(event_traffic_completion_filtfilt, open(event_traffic_completion_filtfilt_file, "wb"))
print("len_before: {}, len_after_filtfilt: {}".format(len_before, len_after))
