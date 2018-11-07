# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:21:54 2017

@author: user
"""

import time
import os
import numpy as np
import cPickle

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
    return time_id_d

t=17568
r=72
c=68
#s_grid=np.zeros((t,r,c))
d_grid=np.zeros((t,r,c))



path="H:/query_beijing_parsed_filtered"
files=os.listdir(path)
kk=0
for file in files:
    f = open(path+"/"+file)
    kk+=1
    print(kk,'day is over')
    for line in f.readlines():
        line=line.split('\t')
        time_id_d=timeid(line)
#        row_id_s=int(line[7])
#        col_id_s=int(line[6])
        row_id_d=int(line[9])
        col_id_d=int(line[8])
        
        if time_id_d<17568*5 and time_id_d>-1:
            time_id_5min=int(time_id_d/5)
            d_grid[time_id_5min][row_id_d][col_id_d] += 1
               

#cPickle.dump(s_grid,open("/home/binbingliao/gyf/s_grid.pkl",'wb'))
cPickle.dump(d_grid, open("H:/d_grid_beijing.pkl",'wb'))
                 
                 
                