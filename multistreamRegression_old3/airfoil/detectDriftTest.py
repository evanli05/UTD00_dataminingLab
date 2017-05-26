# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:20:57 2017

@author: BD, YL
"""

import numpy as np
import scipy as sp
import scipy.stats
import math
import matplotlib.pyplot as plt
y = np.loadtxt('y.txt')
markDrift = np.zeros((1,1501))
ind = 0
temp = 0

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    se = scipy.stats.sem(a)
#    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

#sigmoid
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

width = []
#for i in range(0,1494):
 # d = y[i: i+5]
#  dd = mean_confidence_interval(d, confidence=0.95)
 # width.append(sigmoid(dd[2]-dd[1]))
  #print width

for i in range(0,1449):
  d = y[i: i+50]
#  pre = y[]
  dd = mean_confidence_interval(d, confidence=0.95)
  width.append(sigmoid(dd[2]-dd[1]))


#print width

#check the difference of width, suppo

for diff in range(15, 1449):
    if width[diff] - width[diff-15] < -0.06:
        markDrift[0, diff] = 1

#print markDrift
               
#for ind in markDrift:
#    if markDrift.astype(int) > 0.5:
#        print '1'
#    if markDrift[0, ind].astype(int) > 0.5:
#        print markDrift
                 
       

#for i,x in enumerate(markDrift): 
#    if x[i] > 0:
#        print i
        


print 'the drift index is: '

for pointDrift in np.nditer(markDrift):
    ind = ind + 1
    if pointDrift > 0.5 and temp < ind - 50:
        temp = ind
        print ind
    else:
        continue
#        pointDrift = pointDrift + 50
#        print ind

        
        

plt.plot(width)
plt.ylabel('sigmoid width')
plt.show()
#plt.plot(markDrift, s = 80, facecolors = 'none', edgecolors = 'r')
