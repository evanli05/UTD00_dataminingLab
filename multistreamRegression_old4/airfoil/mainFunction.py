from properties import Properties
from changedetection import ChangeDetection
import numpy as np
import scipy
import math, numpy as np
from scipy.stats import beta, binom
import matplotlib.pyplot as plt
from decimal import Decimal
import sys
from operator import truediv
from mlab.releases import R2013a as mlab
from random import randint
import random
import csv

def __calcBetaDistAlpha(list, sampleMean, sampleVar):
    if sampleMean == -1:
        sampleMean = np.mean(list)
    if sampleVar == -1:
        sampleVar = np.var(list)
    c = (sampleMean * (1 - sampleMean) / sampleVar) - 1
    return sampleMean * c


def __calcBetaDistBeta(list, alphaChange, sampleMean):
    if sampleMean == -1:
        sampleMean = np.mean(list)
    return alphaChange * ((1.0 / sampleMean) - 1)

gamma = 0.5
maxWindowSize = 1000
sensitivity = 0.05
CUSHION = 30

def detectChange(slidingWindow):
    estimatedChangePoint = -1
    N = len(slidingWindow)
    cushion = max(CUSHION, int(math.floor(N ** gamma)))
    if N > maxWindowSize:
            print('Current target Window Size is: ' + str(N) + 'it exceed the max window size')
            return 0
    if N > 2 * cushion and np.mean(slidingWindow[0:N]) <= 0.3:
        print('Current target Window Size is: ' + str(N))
        print(
            'But overall confidence fell below ' + str(Properties.CONFCUTOFF) + ', so update classifier')
        return 0
    threshold = -math.log(sensitivity)
    w = 0.0
    kAtMaxW = -1
    kindex = np.arange(cushion, N - cushion + 1)
    for k in kindex:
        xbar0 = np.mean(slidingWindow[:k])
        var0 = np.var(slidingWindow[:k])
        xbar1 = np.mean(slidingWindow[k:])
        var1 = np.var(slidingWindow[k:])
        if True:
            skn = 0.0
            alphaPreChange = __calcBetaDistAlpha(slidingWindow[:k], xbar0, var0)
            betaPreChange = __calcBetaDistBeta(slidingWindow[:k], alphaPreChange, xbar0)
            alphaPostChange = __calcBetaDistAlpha(slidingWindow[k:], xbar1, var1)
            betaPostChange = __calcBetaDistBeta(slidingWindow[k:], alphaPostChange, xbar1)
            try:
                swin = map(float, slidingWindow[k:])
                swin1 =map(float, slidingWindow[:k])
                denom = [beta.pdf(s, alphaPreChange, betaPreChange) for s in swin]
                nor_denom = np.array([0.001 if h == 0 else h for h in denom])
                nor_swin = swin / nor_denom
                skn = sum([Decimal(beta.pdf(ns, alphaPostChange, betaPostChange)) for ns in nor_swin])
                print k,skn
            except:
                e = sys.exc_info()
                print str(e[1])
                raise Exception('Error in calculating skn')
            if skn > w:
                w = skn
                kAtMaxW = k
    if w >= threshold and kAtMaxW != -1:
        estimatedChangePoint = kAtMaxW
        print('Estimated change point is ' + str(estimatedChangePoint) + ', detected at ' + str(N))

    return estimatedChangePoint


sourceList=[]
targetList=[]
with open('src.csv') as csvfile:
    reader = csv.reader(csvfile)
    #row1=next(reader)
    rowLen=450
    for i in range(0,rowLen):
        sourceList.append([])
    for row in reader:
        for i in range(0,rowLen):
            sourceList[i].append(row[i])
with open('tar.csv') as csvfile:
    reader = csv.reader(csvfile)
    #row1=next(reader)
    rowLen=4050
    for i in range(0,rowLen):
        targetList.append([])
    for row in reader:
        for i in range(0,rowLen):
            targetList[i].append(row[i])

sourceIndex=100
targetIndex=900
source=sourceList[0:100]
target=targetList[0:900]
paras= mlab.removeShift(source,target)
print 'Initialization finished...'
for i in range(1000,1100):
    print ("i is: ",i)
    dataType=randint(1,10)
    if dataType<2:
        print("get data from source")
        source.append(sourceList[sourceIndex])
        sourceIndex+=1
    else:
        print("get data from target")
        xproduct=paras[9]
        print 'paras 9 is: ',paras[9]
        for ss in range(0,5):
            xproduct+=paras[4+ss]*float(targetList[targetIndex][ss])
            print 'xproduct is: ',xproduct

        div=(2*paras[0]*paras[3]+1/paras[2])*(-2*paras[0]*xproduct+(1/paras[2])*paras[1])
        temp=targetList[targetIndex][0:4]
        temp.append(1/div)
        target.append(temp)
        targetIndex+=1
        print 'error not before here'
        #print 'target is' ,target
        target_to_detect=[]
        for ii in range(0,len(target)-1):
            target_to_detect.append(target[ii][5])
        changeIndex = detectChange(target_to_detect)
        print 'error not before here 2'
#        print changeIndex
        if changeIndex!=-1:
            print 'change point is: ', changeIndex
            print 'length of target is: ',len(target)
            newTarget=target[changeIndex:]
            paras= mlab.removeShift(source,newTarget)
            target=newTarget
# 




