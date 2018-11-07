#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

"""
    for each link, extract link info feature
"""

root_dir = "../../Baidu/pkl/"

input_event_top5_link_info_beijing_file = root_dir + "event_link_set_all_beijing_1km_link_info"

output_event_top5_link_info_feature_beijing_file = root_dir + "event_link_set_all_beijing_1km_link_info_feature.pkl"

one_hot_index = [2, 3, 5, 6, 11, 33, 34]
numeric_index = [4, 12, 24, 27]
one_hot_feature_type_len = len(one_hot_index)
numeric_feature_len = len(numeric_index)
# extract link info feature of each link, 44 cols
"""
    mapid Char(8)
    id Char(13)
    kind_num Char(2) # nominal, one-hot
    kind Char(30) # nominal, one-hot
    width Char(3) # numeric, float
    direction Char(1) # nominal, one-hot
    toll Char(1) # nominal, one-hot
    const_st Char(1)
    undconcrid Char(13)
    snodeid Char(13)
    enodeid Char(13)
    pathclass Char(2) # nominal, one-hot
    length Char(8) # numeric, float
    detailcity Char(1)
    through Char(1)
    unthrucrid Char(13)
    ownership Char(1)
    road_cond Char(1)
    special Char(1)
    admincodel Char(6)
    admincoder Char(6)
    uflag Char(1)
    onewaycrid Char(13)
    accesscrid Char(13)
    speedclass Char(1) # numeric, float
    lanenums2e Char(2) 
    lanenume2s Char(2) 
    lanenum Char(1) # numeric, float
    vehcl_type Char(32)
    elevated Char(1)
    structure Char(1)
    usefeecrid Char(13)
    usefeetype Char(1)
    spdlmts2e Char(4) # nominal, one-hot
    spdlmte2s Char(4) # nominal, one-hot
    spdsrcs2e Char(1)
    spdsrce2s Char(1)
    spdms2e Char(1)
    spdme2s Char(1)
    dc_type Char(1)
    verify_flag Char(4)
    walk_form Char(256)
    pre_launch Char(256)
    status Char(4)
"""


def link_info_feature_extraction(link_info_file):
    link_list = list()
    # init the feature dict
    feature_dim_list = list()
    feature_dict_list = list()
    for i in range(one_hot_feature_type_len):
        feature_dim_list.append(0)
        feature_dict_list.append(defaultdict())
    # get the dict
    link_num = 0
    for line in open(link_info_file):
        link_num += 1
        temp_arr = line.strip("\n").replace("\"", "").split("\t")

        link_list.append(temp_arr[1])
        for i in range(one_hot_feature_type_len):
            if temp_arr[one_hot_index[i]] not in feature_dict_list[i]:
                feature_dict_list[i][temp_arr[one_hot_index[i]]] = feature_dim_list[i]
                feature_dim_list[i] += 1
    total_feature_len = 0
    for i in range(one_hot_feature_type_len):
        print("feature_dim_list[{}] is {}".format(one_hot_index[i], feature_dim_list[i]))
        total_feature_len += feature_dim_list[i]
    # one_hot and numeric feature extraction
    one_hot_feature_list = list()
    numeric_feature = np.zeros((link_num, numeric_feature_len), dtype=np.float)
    for i in range(one_hot_feature_type_len):
        one_hot_feature_list.append(np.zeros((link_num, feature_dim_list[i]), dtype=np.float))

    link_num = 0
    for line in open(link_info_file):
        temp_arr = line.strip("\n").replace("\"", "").split("\t")
        for i in range(one_hot_feature_type_len):
            temp_feature = temp_arr[one_hot_index[i]]
            temp_feature_index = feature_dict_list[i][temp_feature]
            one_hot_feature_list[i][link_num, temp_feature_index] = 1
        for i in range(numeric_feature_len):
            temp_feature = temp_arr[numeric_index[i]]
            numeric_feature[link_num, i] = float(temp_feature)

        link_num += 1
    # merge the feature list into one np array
    one_hot_feature_list.append(numeric_feature)
    feature_dim_list.append(numeric_feature_len)
    total_feature_len += numeric_feature_len
    print("total_feature_len is {}".format(total_feature_len))
    link_info_feature = np.zeros((link_num, total_feature_len), dtype=np.float)
    col_index = 0
    for i in range(one_hot_feature_type_len + 1):  # plus numeric feature
        link_info_feature[:, col_index: col_index+feature_dim_list[i]] = one_hot_feature_list[i]
        col_index += feature_dim_list[i]

    return link_list, link_info_feature


link_list, link_info_feature = link_info_feature_extraction(input_event_top5_link_info_beijing_file)
cPickle.dump((link_list, link_info_feature), open(output_event_top5_link_info_feature_beijing_file, "wb"))
