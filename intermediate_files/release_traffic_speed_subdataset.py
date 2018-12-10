#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

root_dir = "../../Baidu/pkl/"
input_traffic_file = root_dir + "event_traffic_beijing_1km_mv_avg_15min_completion.pkl"
input_link_id_hash_map_file = root_dir + "link_id_hash_map"
output_traffic_file = root_dir + "traffic_speed_sub-dataset"
output_traffic_pkl_file = root_dir + "traffic_speed_sub-dataset.pkl"


def traffic_speed_extraction(origin_traffic_pkl_file, link_id_hash_map_file, traffic_pkl_file, traffic_file):
    # load traffic speed
    print("loading traffic speed dataset ...")
    event_traffic = cPickle.load(open(origin_traffic_pkl_file, "rb"))
    total_link_num = len(event_traffic.keys())
    print("link_num is {}".format(total_link_num))
    # load link_id_hash_map
    link_id_hash_map = defaultdict()
    for line in open(link_id_hash_map_file):
        temp_arr = line.strip("\n").split("\t")
        link_id_hash_map[temp_arr[0]] = temp_arr[1]
    # write to pkl file
    '''
    output_traffic = defaultdict()
    for link_id in event_traffic.keys():
        output_traffic[link_id_hash_map[link_id]] = event_traffic[link_id]
    print("dumping traffic speed dataset ...")
    cPickle.dump(output_traffic, open(traffic_pkl_file, "wb"))
    '''
    # write to text file
    print("writing to text file ...")
    fw = open(traffic_file, "wb")
    link_cnt = 0
    for link_id in event_traffic.keys():
        link_cnt += 1
        if link_cnt % 100 == 0:
            print("{}/{}".format(link_cnt, total_link_num))
        new_link_id = link_id_hash_map[link_id]
        temp_len = len(event_traffic[link_id])
        for i in range(temp_len):
            fw.write("{}, {}, {}\n".format(new_link_id, i, event_traffic[link_id][i]))
    fw.close()


if __name__ == '__main__':
    traffic_speed_extraction(input_traffic_file, input_link_id_hash_map_file,
                             output_traffic_pkl_file, output_traffic_file)
