import math as m
import random as r
import sys
from RegDataPoint import DataPoint
import numpy as np
import random

#get y according to original function
def getYAccFirstConcept(indVars, src):
    return DataPoint(indVars, 10*m.sin(m.pi*indVars[0]*indVars[1]) + 20*m.pow((indVars[2]-0.5), 2) + 10*indVars[3] + 5*indVars[4] + r.random(), src)

#get y according to first region
def getYAccFirstRegion(indVars, src):
    return DataPoint(indVars, 10*indVars[0]*indVars[1] + 20*(indVars[2]-0.5) + 10*indVars[3] + 5*indVars[4] + r.random(), src)

#get y according to second region
def getYAccSecondRegion(indVars, src):
    return DataPoint(indVars, 10*m.cos(m.pi*indVars[1]*indVars[2]) + 20*(indVars[1]-0.5) + m.exp(indVars[4]) + 5*m.pow(indVars[4],2) + r.random(), src)

#calculate square of distance between two points
def calcDistSq(point1, point2):
    return np.sum(np.power(point1-point2, 2))

"""
method to gen data
numInd = number of independent variables
n = number of data points
srcSplit = percentage of points that go to the source stream
"""
def genData(outSrcFile, outTrgFile, numInd, n, srcSplit):
    foutSrcFile = open(outSrcFile, "w")
    foutTrgFile = open(outTrgFile, "w")
    numPointsFirstConcept = int(n/4)

    """process first concept"""
    params_1 = np.zeros((numPointsFirstConcept, numInd))
    params_2 = np.zeros((numPointsFirstConcept, numInd))
    count_1 = 0
    count_2 = 0
    pointsFromSingleConcept = []
    for i in range(numPointsFirstConcept):
        r1 = random.random()
        tmp = None
        src = None
        if r1 > 0.5:
            for j in range(numInd):
                if j==0:
                    params_1[count_1, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_1[count_1, j] = np.random.uniform(low=0.7, high=1.0, size=1)
                else:
                    params_1[count_1, j] = np.random.uniform(low=0.0, high=0.3, size=1)
            tmp = params_1[count_1, :]
            src = 0
            count_1 += 1
        else:
            for j in range(numInd):
                if j==0:
                    params_2[count_2, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_2[count_2, j] = np.random.uniform(low=0.0, high=0.3, size=1)
                else:
                    params_2[count_2, j] = np.random.uniform(low=0.7, high=1.0, size=1)
            tmp = params_2[count_2, :]
            src = 1
            count_2 += 1

        pointsFromSingleConcept.append(getYAccFirstConcept(tmp, src))
    params_1 = params_1[:count_1, :]
    params_2 = params_2[:count_2, :]
    meanVals_1 = np.mean(params_1, axis=0)
    meanVals_2 = np.mean(params_2, axis=0)

    distSquares_1 = []
    distSquares_2 = []
    distSquares = []
    for point in pointsFromSingleConcept:
        if point.src == 0:
            distSquares.append(calcDistSq(point.indVars, meanVals_1))
            distSquares_1.append(calcDistSq(point.indVars, meanVals_1))
        else:
            distSquares.append(calcDistSq(point.indVars, meanVals_2))
            distSquares_2.append(calcDistSq(point.indVars, meanVals_2))
    variance_1 = m.pow(np.std(np.sqrt(distSquares_1)), 2)
    variance_2 = m.pow(np.std(np.sqrt(distSquares_2)), 2)
    variance = variance_1 + variance_2


    numSrcPoints = 0
    print("writing the first concept points")
    for i in range(len(distSquares)):
        prob = m.exp(-1*(distSquares[i]/(2*variance)))
        randNum = r.random()
        print randNum , prob, distSquares[i], variance
        if randNum < prob and numSrcPoints<(numPointsFirstConcept*srcSplit):
            foutSrcFile.write(pointsFromSingleConcept[i].toString())
            numSrcPoints += 1
        else:
            foutTrgFile.write(pointsFromSingleConcept[i].toString())


    """ process second concept """
    numPointsSecondConcept = int(n / 4)

    # process second concept
    params_1 = np.zeros((numPointsSecondConcept, numInd))
    params_2 = np.zeros((numPointsSecondConcept, numInd))
    count_1 = 0
    count_2 = 0
    pointsFromSingleConcept = []
    for i in range(numPointsSecondConcept):
        r1 = random.random()
        tmp = None
        src = None
        if r1 > 0.5:
            for j in range(numInd):
                if j == 0:
                    params_1[count_1, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_1[count_1, j] = np.random.uniform(low=0.7, high=1.0, size=1)
                else:
                    params_1[count_1, j] = np.random.uniform(low=0.0, high=0.3, size=1)
            tmp = params_1[count_1, :]
            src = 0
            count_1 += 1

            pointsFromSingleConcept.append(getYAccFirstRegion(tmp, src))
        else:
            for j in range(numInd):
                if j == 0:
                    params_2[count_2, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_2[count_2, j] = np.random.uniform(low=0.0, high=0.3, size=1)
                else:
                    params_2[count_2, j] = np.random.uniform(low=0.7, high=1.0, size=1)
            tmp = params_2[count_2, :]
            src = 1
            count_2 += 1
            pointsFromSingleConcept.append(getYAccFirstConcept(tmp, src))

    params_1 = params_1[:count_1, :]
    params_2 = params_2[:count_2, :]
    meanVals_1 = np.mean(params_1, axis=0)
    meanVals_2 = np.mean(params_2, axis=0)

    distSquares_1 = []
    distSquares_2 = []
    distSquares = []
    for point in pointsFromSingleConcept:
        if point.src == 0:
            distSquares.append(calcDistSq(point.indVars, meanVals_1))
            distSquares_1.append(calcDistSq(point.indVars, meanVals_1))
        else:
            distSquares.append(calcDistSq(point.indVars, meanVals_2))
            distSquares_2.append(calcDistSq(point.indVars, meanVals_2))
    variance_1 = m.pow(np.std(np.sqrt(distSquares_1)), 2)
    variance_2 = m.pow(np.std(np.sqrt(distSquares_2)), 2)
    variance = variance_1 + variance_2

    numSrcPoints = 0
    print("writing the second concept points")
    for i in range(len(distSquares)):
        prob = m.exp(-1 * (distSquares[i] / (2 * variance)))
        randNum = r.random()
        if randNum < prob and numSrcPoints < (numPointsFirstConcept * srcSplit):
            foutSrcFile.write(pointsFromSingleConcept[i].toString())
            numSrcPoints += 1
        else:
            foutTrgFile.write(pointsFromSingleConcept[i].toString())

    """ process third concept """
    numPointsThirdConcept = int(n / 4)

    # process third concept
    params_1 = np.zeros((numPointsThirdConcept, numInd))
    params_2 = np.zeros((numPointsThirdConcept, numInd))
    count_1 = 0
    count_2 = 0
    pointsFromSingleConcept = []
    for i in range(numPointsThirdConcept):
        r1 = random.random()
        tmp = None
        src = None
        if r1 > 0.5:
            for j in range(numInd):
                if j == 0:
                    params_1[count_1, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_1[count_1, j] = np.random.uniform(low=0.7, high=1.0, size=1)
                else:
                    params_1[count_1, j] = np.random.uniform(low=0.0, high=0.3, size=1)
            tmp = params_1[count_1, :]
            src = 0
            count_1 += 1

            pointsFromSingleConcept.append(getYAccFirstConcept(tmp, src))
        else:
            for j in range(numInd):
                if j == 0:
                    params_2[count_2, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_2[count_2, j] = np.random.uniform(low=0.0, high=0.3, size=1)
                else:
                    params_2[count_2, j] = np.random.uniform(low=0.7, high=1.0, size=1)
            tmp = params_2[count_2, :]
            src = 1
            count_2 += 1
            pointsFromSingleConcept.append(getYAccSecondRegion(tmp, src))

    params_1 = params_1[:count_1, :]
    params_2 = params_2[:count_2, :]
    meanVals_1 = np.mean(params_1, axis=0)
    meanVals_2 = np.mean(params_2, axis=0)

    distSquares_1 = []
    distSquares_2 = []
    distSquares = []
    for point in pointsFromSingleConcept:
        if point.src == 0:
            distSquares.append(calcDistSq(point.indVars, meanVals_1))
            distSquares_1.append(calcDistSq(point.indVars, meanVals_1))
        else:
            distSquares.append(calcDistSq(point.indVars, meanVals_2))
            distSquares_2.append(calcDistSq(point.indVars, meanVals_2))
    variance_1 = m.pow(np.std(np.sqrt(distSquares_1)), 2)
    variance_2 = m.pow(np.std(np.sqrt(distSquares_2)), 2)
    variance = variance_1 + variance_2

    numSrcPoints = 0
    print("writing the third concept points")
    for i in range(len(distSquares)):
        prob = m.exp(-1 * (distSquares[i] / (2 * variance)))
        randNum = r.random()
        if randNum < prob and numSrcPoints < (numPointsFirstConcept * srcSplit):
            foutSrcFile.write(pointsFromSingleConcept[i].toString())
            numSrcPoints += 1
        else:
            foutTrgFile.write(pointsFromSingleConcept[i].toString())

    """ process fourth concept """
    numPointsFourthConcept = int(n / 4)

    # process Fourth concept
    params_1 = np.zeros((numPointsFourthConcept, numInd))
    params_2 = np.zeros((numPointsFourthConcept, numInd))
    count_1 = 0
    count_2 = 0
    pointsFromSingleConcept = []
    for i in range(numPointsFourthConcept):
        r1 = random.random()
        tmp = None
        src = None
        if r1 > 0.5:
            for j in range(numInd):
                if j == 0:
                    params_1[count_1, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_1[count_1, j] = np.random.uniform(low=0.7, high=1.0, size=1)
                else:
                    params_1[count_1, j] = np.random.uniform(low=0.0, high=0.3, size=1)
            tmp = params_1[count_1, :]
            src = 0
            count_1 += 1

            pointsFromSingleConcept.append(getYAccFirstRegion(tmp, src))
        else:
            for j in range(numInd):
                if j == 0:
                    params_2[count_2, j] = np.random.uniform(size=1)
                elif j == 3:
                    params_2[count_2, j] = np.random.uniform(low=0.0, high=0.3, size=1)
                else:
                    params_2[count_2, j] = np.random.uniform(low=0.7, high=1.0, size=1)
            tmp = params_2[count_2, :]
            src = 1
            count_2 += 1
            pointsFromSingleConcept.append(getYAccFirstConcept(tmp, src))

    params_1 = params_1[:count_1, :]
    params_2 = params_2[:count_2, :]
    meanVals_1 = np.mean(params_1, axis=0)
    meanVals_2 = np.mean(params_2, axis=0)

    distSquares_1 = []
    distSquares_2 = []
    distSquares = []
    for point in pointsFromSingleConcept:
        if point.src == 0:
            distSquares.append(calcDistSq(point.indVars, meanVals_1))
            distSquares_1.append(calcDistSq(point.indVars, meanVals_1))
        else:
            distSquares.append(calcDistSq(point.indVars, meanVals_2))
            distSquares_2.append(calcDistSq(point.indVars, meanVals_2))
    variance_1 = m.pow(np.std(np.sqrt(distSquares_1)), 2)
    variance_2 = m.pow(np.std(np.sqrt(distSquares_2)), 2)
    variance = variance_1 + variance_2

    numSrcPoints = 0
    print("writing the third concept points")
    for i in range(len(distSquares)):
        prob = m.exp(-1 * (distSquares[i] / (2 * variance)))
        randNum = r.random()
        if randNum < prob and numSrcPoints < (numPointsFirstConcept * srcSplit):
            foutSrcFile.write(pointsFromSingleConcept[i].toString())
            numSrcPoints += 1
        else:
            foutTrgFile.write(pointsFromSingleConcept[i].toString())

genData('RegSynLocalAbruptDrift_source_streamwhole.csv', 'RegSynLocalAbruptDrift_target_streamwhole.csv', 5, 100000, 0.1)
