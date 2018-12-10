# -*- coding: utf-8 -*-

import datetime
import os
import query.mercator_convertor
from collections  import defaultdict
import numpy as np

input_dir = "/home/liaobinbing/Baidu/query_beijing_parsed_filtered/"
output_dir = "/home/liaobinbing/Baidu/query_sub-dataset/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
END_TIME = datetime.datetime.strptime("2017-05-31 23:59:59", "%Y-%m-%d %H:%M:%S")


def get_start_time(query_info):
    loc_time = datetime.datetime.strptime(query_info[0], "%Y-%m-%d %H:%M:%S")
    walk_time = int(query_info[1]) + int(query_info[11])
    start_timestamp = loc_time + datetime.timedelta(minutes=walk_time)
    return start_timestamp


file_list = os.listdir(input_dir)
file_list.sort()
file_cnt = 0
total_file_cnt = len(file_list)
query_cnt = 0
for file_name in file_list:
    temp_file = open(input_dir + file_name)
    temp_out = open(output_dir + file_name, "wb")
    temp_out_str_list = []
    file_cnt += 1
    for line in temp_file.readlines():
        line = line.split('\t')
        start_time = get_start_time(line)
        # invalid start time
        if (start_time - END_TIME).total_seconds() > 0:
            continue
        start_lon, start_lat = query.mercator_convertor.pixel2coord(float(line[2]), float(line[3]))
        des_lon, des_lat = query.mercator_convertor.pixel2coord(float(line[4]), float(line[5]))
        travel_time = line[10]
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        temp_out_str_list.append("{}, {} {}, {} {}, {}".format(start_time_str, start_lon, start_lat, des_lon, des_lat, travel_time))
    cur_query_cnt = len(temp_out_str_list)
    for temp_line in temp_out_str_list:
        temp_out.write(temp_line + "\n")
    temp_out.close()
    query_cnt += cur_query_cnt
    print("processing {}/{}, cur_query_cnt: {}, total_query_cnt: {}".format(file_cnt, total_file_cnt, cur_query_cnt, query_cnt))
