# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:06:33 2017

@author: YL
@version: 0.3
@history:
    

"""

import xlrd
#import scipy as sp
import scipy.stats
import numpy as np
#import matplotlib.pyplot as plt

def detRegressionDrift(setName, confidence = 0.95):
    
    setName = 'dataCDrift.xlsx'
    workbook = xlrd.open_workbook(setName)
    sheet = workbook.sheet_by_index(0)
    data = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in \
             range(sheet.nrows)]
    npdata = np.array(data)
#    print npdata
#    return npdata
    
#    here we start to generate multivariate interval
#    dataShape = npdata.shape

    dataStream = np.zeros((6,1))
    streamMean = np.zeros((1,1501))
    streamLBound = np.zeros((1,1501))
    streamHBound = np.zeros((1,1501))
    streamLBound[0, 0] = -1
    streamHBound[0, 0] = 1
    markDrift = np.zeros((1,1501))
#    withinRange = np.zeros((1,1501))
    
#    i here indicates the pointer of storing Mean, hBound, lBound, and mDrift
    i = 1
#    j here indicates the accumulator of checking whether there are five
#    consecutive numbers that is out of the 95% bound
    j = 0
    
    for col in npdata.T:
        
        column = col.T[:,None]
           
#        print 'col = ', column[5, 0]
#        print 'lBound = ', streamLBound[0, i-1]
#        print 'hBound = ', streamHBound[0, i-1]
        
#        if the new y value is within the bound generated, then continue to
#        update the bound with this new y:
        
        if column[5, 0] > streamLBound[0, i-1] - 0.5 and \
                 column[5, 0] < streamHBound[0, i-1] + 0.5:
            
            dataStream = np.concatenate((dataStream, column), axis = 1)

#            generate the 95% confidence interval        
            a = 1.0 * dataStream
            n = len(a)
            m, se = np.mean(a[5,:], axis = 0), scipy.stats.sem(a[5,:])
            h = se * scipy.stats.t._ppf((1 + confidence) / 2., n-1)
#            print "h =", h
#            print "m =", m
#            print "se =", se
            streamMean[0, i] = m
            streamLBound[0, i] = m-h
            streamHBound[0, i] = m+h
            i = i + 1
            
#        if the new y value is out of the bound generated, we need to check
#        whether there are five consecutive values out of the bound
#        case 1: five consecutive values out of bound, target the first point
#        as drift point, then start to use new bound table
#        case 2: not all five consecutive values out of bound, ~drift point,
#        continue

        else:
#            j here represents how many numbers are there that consecutively
#            of the bound
            j = j + 1
            if j < 5:
                i = i+1
                continue
            
#            checkConsecutive = np.sum([withinRange[0,j-4], withinRange[0,j-3], \
#                                       withinRange[0,j-2], withinRange[0,j-1], \
#                                       withinRange[0,j]]) / 5
#            if  checkConsecutive< 1:
#                i = i+1
#                withinRange[0, j] = 1
#                continue

            else:
                j = 0
                dataStream = np.zeros((6, 1))
                dataStream = np.concatenate((dataStream, column), axis = 1)
                a = 1.0 * dataStream
                n = len(a)
                m, se = np.mean(a[5,:], axis = 0), scipy.stats.sem(a[5,:])
                h = se * scipy.stats.t._ppf((1 + confidence) / 2., n-1)
                streamMean[0, i] = m
                streamLBound[0, i] = m-h
                streamHBound[0, i] = m+h
                markDrift[0, i-5] = 1
                i = i + 1
                
    return streamMean,streamLBound, streamHBound, markDrift


Mean, lBound, hBound, mDrift = detRegressionDrift('dataCDrift.xlsx')

disDrift = np.zeros((1,20))
i = 0
k = 0
mDrift = mDrift.astype(int)
print mDrift

#for index in tShift:
#    
#    if tShift[0, i] == 1:
#        disDrift[0, k] = i
#        k = k+1
#    i = i + 1
#
#print disDrift

#plt.plot(Mean)
#plt.show()

