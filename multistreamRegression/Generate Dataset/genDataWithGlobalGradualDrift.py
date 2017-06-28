import math as m
import random as r
import sys
from RegDataPoint import DataPoint
import numpy as np
import random

#get y according to first concept
def getYAccFirstConcept(indVars):
	return DataPoint(indVars, 10*m.sin(m.pi*indVars[0]*indVars[1]) + 20*m.pow(
            (indVars[2]-0.5), 2) + 10*indVars[3] + 5*indVars[4] + r.random())

#get y according to second concept
def getYAccSecondConcept(indVars):
	return DataPoint(indVars, 10*m.sin(m.pi*indVars[3]*indVars[4]) + 20*m.pow(
            (indVars[1]-0.5), 2) + 10*indVars[0] + 5*indVars[2] + r.random())

#get y according to second concept
def getYAccThirdConcept(indVars):
	return DataPoint(indVars, 10*m.sin(m.pi*indVars[1]*indVars[4]) + 20*m.pow(
            (indVars[3]-0.5), 2) + 10*indVars[2] + 5*indVars[0] + r.random())

#calculate square of distance between two points
def calcDistSq(point1, point2):
	return np.sum(np.power(point1-point2, 2))

options = {1: getYAccFirstConcept,
		   2: getYAccSecondConcept,
		   3: getYAccThirdConcept}

numInConcept = 5000
srcSplit = float(np.random.randint(1,5))/10

"""
method to gen data
numInd = number of independent variables
srcSplit = percentage of points that go to the source stream

each concept contains 10000 of instances
in each concept the drift happens in 10 steps gradually

"""
def genData(outSrcFile, outTrgFile, numInd, numDrift):

	foutSrcFile = open(outSrcFile, "w")
	foutTrgFile = open(outTrgFile, "w")

	"""
	here we generate the first part for the global gradual dataset
	this part is just a smooth function which uses only one functions
	no concept drift in this part
	"""
	# process first concept
	srcSplit = float(np.random.randint(1,5))/10
	numPointsFirstConcept = numInConcept
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

	aMethod = options[1];

	distSquares= []
	for point in pointsFromSingleConcept:
		distSquares.append(calcDistSq(point.indVars, meanVals))
	variance = m.pow(np.std(np.sqrt(distSquares)), 2)

	numSrcPoints = 0
	print "writing No.1 concept points, and the ratio is {}".format(srcSplit)
	for i in range(len(distSquares)):
		prob = m.exp(-1*(distSquares[i]/(2*variance)))
		randNum = r.random()
		if randNum < prob and numSrcPoints<(numPointsFirstConcept*srcSplit):
			foutSrcFile.write(pointsFromSingleConcept[i].toString())
			numSrcPoints += 1
		else:
			foutTrgFile.write(pointsFromSingleConcept[i].toString())

	"""
	Now start mixing the second concept in the data set gradually.
	Start mixing after 50k instances, and complete mixing at 80k instances.
	Therefore, after 80k instances only second concept will be there in the dataset.
	So, start with probability 10% and increase the prob by 10% every 3k instances.
	"""
	for k in range(numDrift-1):
		srcSplit = float(np.random.randint(1,10))/10
		numPoints = numInConcept
		params = np.zeros((numPoints, numInd))
		pointsFromSingleConcept = []
		probFromSecondConcept = 0.0
		for j in range(numInd):
			if j == 0:
				params[:, j] = np.random.uniform(size=numPoints)
			elif j == 3:
				params[:, j] = np.random.uniform(low=0.7, high=1.0, size=numPoints)
			else:
				params[:, j] = np.random.uniform(low=0.0, high=0.3, size=numPoints)
		rn = random.randint(1,3)
		bMethod = options[rn]
		for i in range(numPoints):
			if i%(numInConcept/10) == 0:
				probFromSecondConcept += 0.1
			fromSecond = random.random() < probFromSecondConcept
			if fromSecond:
				pointsFromSingleConcept.append(bMethod(params[i, :]))
			else:
				pointsFromSingleConcept.append(aMethod(params[i, :]))
		meanVals = np.mean(params, axis=0)
		aMethod = options[rn]

		distSquares = []
		for point in pointsFromSingleConcept:
			distSquares.append(calcDistSq(point.indVars, meanVals))
		variance = m.pow(np.std(np.sqrt(distSquares)), 2)

		numSrcPoints = 0
		print "writing No.{} concept points, and the ratio is {}".format(k+2, srcSplit)
		for i in range(len(distSquares)):
			prob = m.exp(-1 * (distSquares[i] / (2 * variance)))
			randNum = r.random()
			if randNum < prob and numSrcPoints < (numPoints * srcSplit):
				foutSrcFile.write(pointsFromSingleConcept[i].toString())
				numSrcPoints += 1
			else:
				foutTrgFile.write(pointsFromSingleConcept[i].toString())

	foutSrcFile.close()
	foutTrgFile.close()


genData('RegSynGlobalGradualDrift_source_streamhalf.csv', 'RegSynGlobalGradualDrift_target_streamhalf.csv', 5, 5)
