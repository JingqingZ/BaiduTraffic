# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:28:07 2017

@author: user
"""

import cPickle
import numpy as np


d=cPickle.load(open("H:/d_grid_beijing.pkl",'rb'))

t=17568
r=72
c=68

out=[]

d_sum=np.zeros((61,12,r,c))

for i in range(0,t):
    day=int(i/288)
    day_time=i-day*288
    hour=int(day_time/24)
    for j in range(0,r):
        for k in range(0,c):
            d_sum[day][hour][j][k] +=d[i][j][k]

print('d_sum')            

for i in range(7,61):
    for j in range(12):
        for k in range(r):
            for h in range(c):
                t1=d_sum[i][j][k][h] 
                t2=d_sum[i-7][j][k][h]
                ttt=t1-t2
                tt=1
                if t2!=0:
                    tt=float(t1)/float(t2)
                
                if ttt>300 and tt>0.2:
                    out.append([i,j,k,h])
                    
event_list=[]
n=len(out)
for i in range(0,n):
    rr=out[i][2]
    cc=out[i][3]
    a=out[i][0]
    b=out[i][1]
    tt=a*288+(b*2+1)*12
    start_t=tt
    end_t=tt
    dt=288*7
    if start_t-dt>-1:
        while d[start_t][rr][cc]-1.1*d[start_t-dt][rr][cc]>0:
            if start_t-dt>0:
                start_t=start_t-1
            else:
                break
        
    if end_t-dt>-1:
        while d[end_t][rr][cc]-1.1*d[end_t-dt][rr][cc]>0:
            if end_t<87840/5-1:
                end_t=end_t+1
            else:
                break
    

        
    event_list.append([start_t+1,end_t-1,rr,cc])
    
    
event_U=np.zeros((t,r,c))
for i in range(0,n):
    st=event_list[i][0]
    en=event_list[i][1]
    rr=event_list[i][2]
    cc=event_list[i][3]
    for j in range(st,en+1):
        event_U[j][rr][cc]=1
    

out1=[]          
for j in range(0,r):
    for k in range(0,c):
        for i in range(0,t-1):
            if event_U[i][j][k]==0 and event_U[i+1][j][k] ==1:
                start=i+1
                ii=i+1
                while event_U[ii][j][k]==1:
                    ii+=1
                    if ii==t:
                        break
    
                    
                end=ii-1
                
                out1.append([start,end,j,k])

                  
f=open("H:/event_beijing.txt",'wb')
n=len(out1)
print(n)
for i in range(0,n):
    f.write(str(out1[i][0]))
    f.write('\t')
    f.write(str(out1[i][1]))
    f.write('\t')
    f.write(str(out1[i][2]))
    f.write('\t')
    f.write(str(out1[i][3]))
    f.write('\n')
    
f.close()