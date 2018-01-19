#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

root_dir = "F:/"
input_event_top5_link_list_beijing_file = root_dir + "event_top5_link_list.pkl"
haidian_coord_file = root_dir + "beijing_road_mercator"

output_event_link_set_beijing_file = root_dir + "event_top5_link_set_beijing"

print "loading beijing link coord ..."
coord_dict = defaultdict()
link_cnt = 0
for line in open(haidian_coord_file):
    link_cnt += 1
    temp_arr = line.split("\t")
    coord_dict[temp_arr[0]] = temp_arr[1]+"\t"+temp_arr[2].strip("\n")
print "link_cnt is {}".format(link_cnt)

len_event = 0
print "loading event top5 link list ..."
event_link_list = cPickle.load(open(input_event_top5_link_list_beijing_file, "rb"))
len_event = len(event_link_list)
print "event_link_list len is {}".format(len_event)

event_link_set_beijing = set()
for i in range(len_event):
    for temp_link in event_link_list[i]:
        event_link_set_beijing.add(temp_link)

len_event_link_set = len(event_link_set_beijing)
print "event_link_set_beijing len is {}".format(len_event_link_set)

fw = open(output_event_link_set_beijing_file, "wb")
for ele in event_link_set_beijing:
    fw.write(ele+"\t"+coord_dict[ele]+"\n")

fw.close()
print "writing completed."
