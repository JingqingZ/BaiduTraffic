# -*- coding: utf-8 -*-

from math import sqrt, exp
from collections  import defaultdict
import numpy as np
import cPickle
import time
from multiprocessing import Pool


# calculate distance between link and query segment
def distance(d_r, d_c, s_r, s_c, lu_r, lu_c):
    cross=(lu_r-d_r)*(s_r-d_r)+(lu_c-d_c)*(s_c-d_c)
    if cross <=0:
        return sqrt((lu_r-d_r)*(lu_r-d_r)+(lu_c-d_c)*(lu_c-d_c))
    else:
        d2=(s_r-d_r)*(s_r-d_r)+(s_c-d_c)*(s_c-d_c)
        if cross>=d2:
            return sqrt((lu_r-s_r)*(lu_r-s_r)+(lu_c-s_c)*(lu_c-s_c))
        else:
            d1=(lu_r-d_r)*(lu_r-d_r)+(lu_c-d_c)*(lu_c-d_c)
            r=cross*cross/d2/d1
            return sqrt(d1*(1-r))


root_dir = "../../Baidu/pkl/"

input_event_beijing_1152_file = root_dir + "event_beijing_extended_filtered.txt"
input_event_link_coord_beijing_1km_file = root_dir + "event_link_set_all_beijing_1km_filtered"
input_query_beijing_all_info_file = root_dir + "query_beijing_0201_all_information.txt"

output_query_distribution_beijing_1km_file = root_dir + "query_distribution_beijing_1km_k_{}.pkl"

MIN_Y = 4792883.01
MIN_X = 12924083.26
MAX_Y = 4864389.46
MAX_X = 12992448.86
GRID_X = 68
GRID_Y = 72
TOTAL_TIME = 61 * 24 * 4  # 5856
impact_factor = 50

print("loading event_beijing_extended_filtered ...")
event_list = list()
for line in open(input_event_beijing_1152_file):
    temp_arr = line.strip("\n").split('\t')
    start_t = int(float(temp_arr[0])/3)
    end_t = int(float(temp_arr[1])/3)
    row_d = int(temp_arr[2])
    col_d = int(temp_arr[3])
    event_list.append((start_t, end_t, row_d, col_d))
event_len = len(event_list)
print("event_len is {}".format(len(event_list)))

print("loading event_link_set_all_beijing_1km_filtered ...")
all_link_set = set()
link_grid_dict = defaultdict()
for line in open(input_event_link_coord_beijing_1km_file):
    temp_arr = line.strip("\n").split("\t")
    all_link_set.add(temp_arr[0])
    coord_x = float(temp_arr[1])
    coord_y = float(temp_arr[2])
    temp_grid_x = int((coord_x - MIN_X) / (MAX_X - MIN_X) * GRID_X)
    temp_grid_y = int((coord_y - MIN_Y) / (MAX_Y - MIN_Y) * GRID_Y)
    if GRID_X > temp_grid_x >= 0 and GRID_Y > temp_grid_y >= 0:
        if (temp_grid_x, temp_grid_y) not in link_grid_dict:
            link_grid_dict[(temp_grid_x, temp_grid_y)] = set()
        link_grid_dict[(temp_grid_x, temp_grid_y)].add((temp_arr[0], coord_x, coord_y))
link_grid_dict_len = len(link_grid_dict.keys())
all_link_len = len(all_link_set)
print("all_link_len is {}".format(all_link_len))
print("link_grid_dict_len is {}".format(link_grid_dict_len))

print("loading input_query_beijing_all_info_file ...")
query_info = list()
for line in open(input_query_beijing_all_info_file):
    query_info.append(line)
query_len = len(query_info)
print("query_len is {}".format(query_len))
print("loading completed.")

# key=link_id, value = np.array((5856, ))
# Init query_distribution_dict
non_zero_cnt = 0
query_distribution_dict = defaultdict()
for link_id in all_link_set:
    query_distribution_dict[link_id] = np.zeros((TOTAL_TIME, ), dtype=np.float)

cnt = 0
while cnt < query_len:
    temp_arr = query_info[cnt].strip("\n").split('\t')
    cnt += 1
    t_d = int(int(temp_arr[1]) / 3)
    if t_d >= TOTAL_TIME or t_d < 0:
        continue
    d_r_id = int(temp_arr[2])
    d_c_id = int(temp_arr[3])
    d_r = float(temp_arr[4])
    d_c = float(temp_arr[5])
    s_r = float(temp_arr[6])
    s_c = float(temp_arr[7])
    for temp_row in range(d_r_id - 1, d_r_id + 2):
        for temp_col in range(d_c_id - 1, d_c_id + 2):
            if (temp_row, temp_col) in link_grid_dict:
                for link in link_grid_dict[(temp_row, temp_col)]:
                    link_to_query_dis = distance(d_r, d_c, s_r, s_c, link[1], link[2])
                    query_distribution_dict[link[0]][t_d] += exp(-link_to_query_dis / impact_factor)
                    non_zero_cnt += 1

print("non_zero_cnt is {}".format(non_zero_cnt))

cPickle.dump(query_distribution_dict, open(output_query_distribution_beijing_1km_file.format(impact_factor), 'wb'))
