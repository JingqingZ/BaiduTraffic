#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np

root_dir = "F:/"
input_event_link_set_beijing_file = root_dir + "event_top5_link_set_beijing"
beijing_link_info_file = root_dir + "beijing roadnet/R.mid"

output_event_link_info_beijing_file = root_dir + "event_top5_link_info_beijing"

print "loading beijing link coord ..."
link_info_dict = defaultdict()
link_cnt = 0
for line in open(beijing_link_info_file):
    link_cnt += 1
    temp_arr = line.split("\t")
    temp_link_id = temp_arr[1].strip("\"")
    link_info_dict[temp_link_id] = line
print "link_cnt is {}".format(link_cnt)

fw = open(output_event_link_info_beijing_file, "wb")
for line in open(input_event_link_set_beijing_file):
    temp_arr = line.split("\t")
    fw.write(link_info_dict[temp_arr[0]])
fw.close()

print "writing completed."
