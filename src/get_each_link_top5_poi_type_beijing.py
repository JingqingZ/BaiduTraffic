#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np


root_dir = "F:/"
input_beijing_poi_file = root_dir+"beijing_poi"
input_event_link_set_beijing_file = root_dir + "event_top5_link_set_beijing"

output_poi_type_coarse_file = root_dir+"each_link_top5_poi_type_coarse_beijing"
output_poi_type_fine_file = root_dir + "each_link_top5_poi_type_fine_beijing"

POI_NUM = 5
MIN_Y = 4792883.01
MIN_X = 12924083.26
MAX_Y = 4864389.46
MAX_X = 12992448.86

GRID_X = 68
GRID_Y = 72

print "loading poi ..."
poi_id = list()
poi_name = defaultdict()
poi_alias_name = defaultdict()
poi_coord = list()
poi_grid = defaultdict()
poi_type_coarse = list()
poi_type_fine = list()
coarse_query_type = defaultdict()
fine_query_type = defaultdict()
poi_cnt = 0
coarse_query_type_cnt = 0
fine_query_type_cnt = 0
for line in open(input_beijing_poi_file):
    temp_arr = line.split("\t")
    poi_id.append(temp_arr[2])
    poi_name[temp_arr[3]] = poi_cnt
    alias_name_arr = temp_arr[4].split("$")
    for temp_alias_name in alias_name_arr:
        poi_alias_name[temp_alias_name] = poi_cnt
    poi_coord.append((temp_arr[11], temp_arr[12]))

    temp_grid_x = int((float(temp_arr[11]) - MIN_X) / (MAX_X - MIN_X) * GRID_X)
    temp_grid_y = int((float(temp_arr[12]) - MIN_Y) / (MAX_Y - MIN_Y) * GRID_Y)
    if temp_grid_x >=0 and temp_grid_x < GRID_X and temp_grid_y >=0 and temp_grid_y < GRID_Y:
        if (temp_grid_x, temp_grid_y) not in poi_grid:
            poi_grid[(temp_grid_x, temp_grid_y)] = list()
        poi_grid[(temp_grid_x, temp_grid_y)].append((poi_cnt, temp_arr[11], temp_arr[12]))

    type_arr = temp_arr[23].split(";")
    # print temp_arr[23]
    poi_type_coarse.append(type_arr[0])
    if type_arr[0] not in coarse_query_type:
        coarse_query_type[type_arr[0]] = coarse_query_type_cnt
        coarse_query_type_cnt += 1
    temp_fine_query_type = type_arr[0]
    if len(type_arr) > 1:
        temp_fine_query_type = type_arr[1]
    poi_type_fine.append(temp_fine_query_type)
    if temp_fine_query_type not in fine_query_type:
        fine_query_type[temp_fine_query_type] = fine_query_type_cnt
        fine_query_type_cnt += 1
    poi_cnt += 1
print "poi_cnt is {},\tcoarse_query_type_cnt is {},\tfine_query_type_cnt is {}"\
    .format(poi_cnt, coarse_query_type_cnt, fine_query_type_cnt)


def get_top5_poi_index_from_coord(coord_x, coord_y):
    float_x = float(coord_x)
    float_y = float(coord_y)
    temp_grid_x = int((float_x - MIN_X) / (MAX_X - MIN_X) * GRID_X)
    temp_grid_y = int((float_y - MIN_Y) / (MAX_Y - MIN_Y) * GRID_Y)
    index_list = list()
    index_distance = defaultdict()
    for temp_poi in poi_grid[(temp_grid_x, temp_grid_y)]:
        temp_x = float(temp_poi[1])
        temp_y = float(temp_poi[2])
        temp_dist = math.sqrt(math.pow(temp_x-float_x, 2)+math.pow(temp_y-float_y, 2))
        if temp_dist < 500:
            index_distance[temp_poi[0]] = temp_dist
    # find the nearest 5 poi
    temp_list = sorted(index_distance.items(), lambda x, y: cmp(x[1],y[1]))
    for i in range(POI_NUM):
        index_list.append(temp_list[i][0])

    return index_list


fw_coarse = open(output_poi_type_coarse_file, "wb")
fw_fine = open(output_poi_type_fine_file, "wb")
link_cnt = 0
for line in open(input_event_link_set_beijing_file):
    link_cnt += 1
    print "processing {} ...".format(link_cnt)
    temp_arr = line.split("\t")
    temp_index_list = get_top5_poi_index_from_coord(temp_arr[1], temp_arr[2])
    fw_coarse.write(temp_arr[0]+"\t")
    fw_fine.write(temp_arr[0]+"\t")
    for i in range(POI_NUM):
        fw_coarse.write(poi_id[temp_index_list[i]] + "\t" + poi_type_coarse[temp_index_list[i]])
        fw_fine.write(poi_id[temp_index_list[i]] + "\t" + poi_type_fine[temp_index_list[i]])
        if i == POI_NUM - 1:
            fw_coarse.write("\n")
            fw_fine.write("\n")
        else:
            fw_coarse.write("\t")
            fw_fine.write("\t")

fw_coarse.close()
fw_fine.close()

