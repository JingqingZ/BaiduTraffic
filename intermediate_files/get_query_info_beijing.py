# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:56:22 2017

@author: user
"""

import time
import os
from collections  import defaultdict
import numpy as np

def timeid(input1):
    t1=time.strptime(input1[0],"%Y-%m-%d %H:%M:%S")
    mon=t1.tm_mon
    day=t1.tm_mday
    hour=t1.tm_hour
    minute=t1.tm_min
    time_id=60*(24*(30*(mon-4)+day-1)+hour)+minute
#    if time_id >=87840:
#        print"%d\t%d\t%d\t%d"(mon,day,hour,minute)
    time_id_s=time_id+int(input1[1])+int(input1[11])
    time_id_d=time_id_s+int(input1[10])
    time_id_s=int(time_id_s/5)
    time_id_d=int(time_id_d/5)
    return time_id_s,time_id_d


path1="/home/liaobinbing/Baidu/pkl/event_beijing_extended_filtered.txt"
n=0
query = set()
f = open(path1,"rb")
for line in f.readlines():
    line=line.split('\t')
    start_t=int(line[0])
    end_t=int(line[1])
    row_d=int(line[2])
    col_d=int(line[3])
    n+=1
    for temp_row in range(row_d-1, row_d+2):
        for temp_col in range(col_d-1, col_d+2):
            if 0<=temp_row<68 and 0<=temp_col<72:
                query.add((temp_row, temp_col))
print("query_grid num is {}".format(len(query)))

event=[]
path2="/home/liaobinbing/Baidu/query_beijing_parsed_filtered"
files=os.listdir(path2)
all_query=0
event_query=0
kk=0
for file in files:
    f = open(path2+"/"+file)
    kk+=1
    for line in f.readlines():
        all_query+=1
        line=line.split('\t')
        time_id_s,time_id_d=timeid(line)
        row_id_d=int(line[9])
        col_id_d=int(line[8])
        row_d=float(line[5])
        col_d=float(line[4])
        row_s=float(line[3])
        col_s=float(line[2])
        if -1 < time_id_d < 87840:
            if (row_id_d, col_id_d) in query:
                event_query+=1
                event.append([time_id_s,time_id_d,row_id_d,col_id_d,row_d,col_d,row_s,col_s])
    print("processing {}, event_query/all_query: {}/{}".format(kk-1, event_query, all_query))
print(all_query)
print(event_query)
           
path3="/home/liaobinbing/Baidu/pkl/query_beijing_0201_all_information.txt"
f=open(path3,'wb')
n=len(event)
f.write(str(all_query))
f.write('\t')
f.write(str(event_query))
f.write('\n')
for i in range(n):
    f.write(str(event[i][0]))
    f.write('\t')
    f.write(str(event[i][1]))
    f.write('\t')
    f.write(str(event[i][2]))
    f.write('\t')
    f.write(str(event[i][3]))
    f.write('\t')
    f.write(str(event[i][4]))
    f.write('\t')
    f.write(str(event[i][5]))
    f.write('\t')
    f.write(str(event[i][6]))
    f.write('\t')
    f.write(str(event[i][7]))
    f.write('\n')

f.close()                
