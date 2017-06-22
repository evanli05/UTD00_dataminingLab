import math as m
import random as r
import sys
from RegDataPoint import DataPoint
import numpy as np
import random

#get y according to first concept
def getYAccFirstConcept(indVars):
	return DataPoint(indVars, 10*m.sin(m.pi*indVars[0]*indVars[1]) + 20*m.pow((indVars[2]-0.5), 2) + 10*indVars[3] + 5*indVars[4] + r.random())

#get y according to second concept
def getYAccSecondConcept(indVars):
	return DataPoint(indVars, 10*m.sin(m.pi*indVars[3]*indVars[4]) + 20*m.pow((indVars[1]-0.5), 2) + 10*indVars[0] + 5*indVars[2] + r.random())

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
	numPointsFirstConcept = int(n/2)
	# process first concept
	params = np.zeros((numPointsFirstConcept, numInd))
	pointsFromSingleConcept = []
	for j in range(numInd):
		if j==0:
			params[:,j] = np.random.uniform(size=numPointsFirstConcept)
		elif j == 3:
			params[:, j] = np.random.uniform(low=0.7, high=1.0, size=numPointsFirstConcept)
		else:
			params[:, j] = np.random.uniform(low=0.0, high=0.3, size=numPointsFirstConcept)

	for i in range(numPointsFirstConcept):
		pointsFromSingleConcept.append(getYAccFirstConcept(params[i, :]))
	meanVals = np.mean(params, axis=0)

	distSquares= []
	for point in pointsFromSingleConcept:
		distSquares.append(calcDistSq(point.indVars, meanVals))
	variance = m.pow(np.std(np.sqrt(distSquares)), 2)

	numSrcPoints = 0
	print("writing the first concept points")
	for i in range(len(distSquares)):
		prob = m.exp(-1*(distSquares[i]/(2*variance)))
		print prob, distSquares[i], variance
		randNum = r.random()
		if randNum < prob and numSrcPoints<(numPointsFirstConcept*srcSplit):
			foutSrcFile.write(pointsFromSingleConcept[i].toString())
			numSrcPoints += 1
		else:
			foutTrgFile.write(pointsFromSingleConcept[i].toString())

	#process second concept
	numPointsSecondConcept = n - numPointsFirstConcept
	params = np.zeros((numPointsSecondConcept, numInd))
	pointsFromSingleConcept = []
	for j in range(numInd):
		if j == 0:
			params[:, j] = np.random.uniform(size=numPointsSecondConcept)
		elif j == 3:
			params[:, j] = np.random.uniform(low=0.7, high=1.0, size=numPointsSecondConcept)
		else:
			params[:, j] = np.random.uniform(low=0.0, high=0.3, size=numPointsSecondConcept)

	for i in range(numPointsSecondConcept):
		pointsFromSingleConcept.append(getYAccSecondConcept(params[i, :]))
	meanVals = np.mean(params, axis=0)

	distSquares = []
	for point in pointsFromSingleConcept:
		distSquares.append(calcDistSq(point.indVars, meanVals))
	variance = m.pow(np.std(np.sqrt(distSquares)), 2)

	numSrcPoints = 0
	print("writing the second concept points")
	for i in range(len(distSquares)):
		prob = m.exp(-1 * (distSquares[i] / (2 * variance)))
		randNum = r.random()
		if randNum < prob and numSrcPoints < (numPointsSecondConcept * srcSplit):
			foutSrcFile.write(pointsFromSingleConcept[i].toString())
			numSrcPoints += 1
		else:
			foutTrgFile.write(pointsFromSingleConcept[i].toString())

	foutSrcFile.close()
	foutTrgFile.close()


genData('RegSynLocalAbruptDrift_source_stream.csv', 'RegSynLocalAbruptDrift_target_stream.csv', 5, 10000, 0.1)
