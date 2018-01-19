#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

"""
    for each link, extract bow feature of poi type
"""

root_dir = "F:/"

input_poi_type_coarse_file = root_dir + "each_link_top5_poi_type_coarse_beijing"
input_poi_type_fine_file = root_dir + "each_link_top5_poi_type_fine_beijing"

output_poi_type_feature_coarse_file = root_dir + "each_link_top5_poi_type_feature_coarse_beijing.pkl"
output_poi_type_feature_fine_file = root_dir + "each_link_top5_poi_type_feature_fine_beijing.pkl"


# extract bow feature of poi type
def poi_type_feature_extraction(poi_type_file):
    link_list = list()
    feature_dim = 0
    feature_dict = defaultdict()
    link_num = 0
    for line in open(poi_type_file):
        link_num += 1
        temp_arr = line.strip("\n").split("\t")
        link_list.append(temp_arr[0])
        for i in range(2, 11, 2):
            if temp_arr[i] not in feature_dict:
                feature_dict[temp_arr[i]] = feature_dim
                feature_dim += 1

    print("link_num is {}, feature_dim is {}".format(link_num, feature_dim))
    """
    link_num is 1275, feature_dim is 22(coarse)
    link_num is 1275, feature_dim is 129(fine)
    """
    poi_type_feature = np.zeros((link_num, feature_dim), dtype=np.float)
    link_num = 0
    for line in open(poi_type_file):
        temp_arr = line.strip("\n").split("\t")
        for i in range(2, 11, 2):
            poi_type_feature[link_num, feature_dict[temp_arr[i]]] = 1
        link_num += 1
    return link_list, poi_type_feature


link_list_coarse, poi_type_feature_coarse = poi_type_feature_extraction(input_poi_type_coarse_file)
link_list_fine, poi_type_feature_fine = poi_type_feature_extraction(input_poi_type_fine_file)

cPickle.dump((link_list_coarse, poi_type_feature_coarse), open(output_poi_type_feature_coarse_file, "wb"))
cPickle.dump((link_list_fine, poi_type_feature_fine), open(output_poi_type_feature_fine_file, "wb"))
