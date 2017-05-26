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

y = np.loadtxt('y.txt')

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

#sigmoid
def sigmoid(x):
  return 1 / (1+math.exp(-x))

width = []
conf = []
#for i in range(0,1494):
 # d = y[i: i+5]
#  dd = mean_confidence_interval(d, confidence=0.95)
 # width.append(sigmoid(dd[2]-dd[1]))
  #print width

for i in range(0,1449):
  d = y[i: i+50]
  dd = mean_confidence_interval(d, confidence=0.95)
  width.append(dd[2]-dd[1])
  conf.append(sigmoid(dd[2]-dd[1]))

#save in a file
file_width_conf = open('file_width_conf', 'w')
"""
file_width_conf.write("Printing the width:")
for item in width:
    print>>file_width_conf, item

file_width_conf.write("Printing the conf:")
for item in conf:
    print>>file_width_conf, item

file_width_conf.write("Printing the pairs:")
"""
for item in zip(width, conf):
    print>>file_width_conf, item
#changepoint = ChangeDetection(width)


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

    # If mean confidence fall below 0.3, must retrain the classifier, so return a changepoint
    if N > maxWindowSize:
            print('Current target Window Size is: ' + str(N) + 'it exceed the max window size')
            return 0
    if N > 2 * cushion and np.mean(slidingWindow[0:N]) <= 0.3:
        print('Current target Window Size is: ' + str(N))
        print(
            'But overall confidence fell below ' + str(Properties.CONFCUTOFF) + ', so update classifier')
        return 0

    threshold = -math.log(sensitivity)
    #print threshold
    w = 0.0
    kAtMaxW = -1

    kindex = np.arange(cushion, N - cushion + 1)
    for k in kindex:
        xbar0 = np.mean(slidingWindow[:k])
        var0 = np.var(slidingWindow[:k])
        xbar1 = np.mean(slidingWindow[k:])
        var1 = np.var(slidingWindow[k:])

        #if xbar1 <= 0.99 * xbar0:
        if True:
            skn = 0.0
            alphaPreChange = __calcBetaDistAlpha(slidingWindow[:k], xbar0, var0)
            betaPreChange = __calcBetaDistBeta(slidingWindow[:k], alphaPreChange, xbar0)
            alphaPostChange = __calcBetaDistAlpha(slidingWindow[k:], xbar1, var1)
            betaPostChange = __calcBetaDistBeta(slidingWindow[k:], alphaPostChange, xbar1)
            #print alphaPreChange


            try:
                swin = map(float, slidingWindow[k:])
                swin1 =map(float, slidingWindow[:k])
                #print swin[1]
                denom = [beta.pdf(s, alphaPreChange, betaPreChange) for s in swin]
                #print denom
                nor_denom = np.array([0.001 if h == 0 else h for h in denom])
                nor_swin = swin / nor_denom
                #print nor_swin
                skn = sum([Decimal(beta.pdf(ns, alphaPostChange, betaPostChange)) for ns in nor_swin])
                print k,skn
                #ne = [beta.pdf(q, alphaPostChange, betaPostChange) for q in swin1]
                #print ne
                #result = []

                # get last index for the lists for iteration
                #end_index = len(nor_denom)
                #print ne[11]/nor_denom[11]
               # for i in range(end_index):
                #    result.append(ne[i] / nor_denom[i])
                #print ratio
            except:
                e = sys.exc_info()
                print str(e[1])
                raise Exception('Error in calculating skn')

            if skn > w:
                w = skn
                kAtMaxW = k
                #print k

    if w >= threshold and kAtMaxW != -1:
        estimatedChangePoint = kAtMaxW
        print('Estimated change point is ' + str(estimatedChangePoint) + ', detected at ' + str(N))

    return estimatedChangePoint


list1 = conf[900:1100]
sampleMean = -1
sampleVar = -1
alpha = __calcBetaDistAlpha(list1, sampleMean, sampleVar)
#beta = __calcBetaDistBeta(list1, sampleMean, sampleVar)

change = detectChange(list1)
print change





