#!/usr/bin/python
# coding=utf-8
import cPickle
from scipy import signal
from collections import defaultdict
import numpy as np

root_dir = "../../data/"
query_distribution_beijing_1km_file = root_dir + "query_distribution_beijing_1km_k_{}.pkl"
query_distribution_beijing_1km_filtfilt_file = root_dir + "query_distribution_beijing_1km_k_{}_filtfilt.pkl"
impact_factor = 300

print("loading query_distribution_beijing_1km_file ...")
query_distribution = cPickle.load(open(query_distribution_beijing_1km_file.format(impact_factor), "rb"))
total_link_len = len(query_distribution.keys())

query_distribution_filtfilt = defaultdict()
# a is a hyper-parameter, the bigger, the smoother
b, a = signal.butter(8, 0.2)

cnt = 0
for link_id in query_distribution.keys():
    if cnt % 1000 == 0:
        print("processing {}/{}".format(cnt, total_link_len))
    temp_query_list = query_distribution[link_id]
    if len(temp_query_list) >=24:
        new_temp_query_list = signal.filtfilt(b, a, temp_query_list, padlen=23)
        query_distribution_filtfilt[link_id] = new_temp_query_list

print("dumping query_distribution_beijing_1km_filtfilt_file ...")
cPickle.dump(query_distribution_filtfilt, open(query_distribution_beijing_1km_filtfilt_file.format(impact_factor), "wb"))
