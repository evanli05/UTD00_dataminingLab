# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:54:45 2017

@author: YL
"""

import numpy as np
#import scipy as sp
#import scipy.stats


a = np.arange(20).reshape((5,4))
data = np.array(a)
n = data.shape[1]

#arrayA = np.array([[1,2], [3,4]])
#arrayB = np.array([[5,6]])
#
#arrayC = np.concatenate((arrayA, arrayB.T), axis =1)

dataStream = np.zeros((5,1))

for col in data.T:
    column = col.T[:,None]
#    print np.shape(col)
    dataStream = np.concatenate((dataStream, column), axis = 1)
    print dataStream
    
