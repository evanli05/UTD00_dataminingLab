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

#get y according to second concept
def getYAccThirdConcept(indVars):
	return DataPoint(indVars, 10*m.sin(m.pi*indVars[1]*indVars[4]) + 20*m.pow((indVars[3]-0.5), 2) + 10*indVars[2] + 5*indVars[0] + r.random())

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
	#first 50k points are from the first concept
	numPointsFirstConcept = 15000
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
	numPointsSecondConcept = 9000
	params = np.zeros((numPointsSecondConcept, numInd))
	pointsFromSingleConcept = []
	probFromSecondConcept = 0.0
	for j in range(numInd):
		if j == 0:
			params[:, j] = np.random.uniform(size=numPointsSecondConcept)
		elif j == 3:
			params[:, j] = np.random.uniform(low=0.7, high=1.0, size=numPointsSecondConcept)
		else:
			params[:, j] = np.random.uniform(low=0.0, high=0.3, size=numPointsSecondConcept)

	for i in range(numPointsSecondConcept):
		if i%3000 == 0:
			probFromSecondConcept += 0.1
		fromSecond = random.random() < probFromSecondConcept
		if fromSecond:
			pointsFromSingleConcept.append(getYAccSecondConcept(params[i, :]))
		else:
			pointsFromSingleConcept.append(getYAccFirstConcept(params[i, :]))
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

	"""
	Now start mixing the third concept (mixed second and third functions) in the data set gradually.
	Start mixing after 80k instances, and complete mixing at 120k instances.
	Therefore, after 120k instances only second concept will be there in the dataset.
	So, start with probability 10% and increase the prob by 10% every 4k instances.
	"""
	numPointsThirdConcept = 12000
	params = np.zeros((numPointsThirdConcept, numInd))
	pointsFromSingleConcept = []
	probFromThirdConcept = 0.0
	for j in range(numInd):
		if j == 0:
			params[:, j] = np.random.uniform(size=numPointsThirdConcept)
		elif j == 3:
			params[:, j] = np.random.uniform(low=0.7, high=1.0, size=numPointsThirdConcept)
		else:
			params[:, j] = np.random.uniform(low=0.0, high=0.3, size=numPointsThirdConcept)

	for i in range(numPointsThirdConcept):
		if i % 4000 == 0:
			probFromThirdConcept += 0.1
		fromThird = random.random() < probFromThirdConcept
		if fromThird:
			pointsFromSingleConcept.append(getYAccThirdConcept(params[i, :]))
		else:
			pointsFromSingleConcept.append(getYAccSecondConcept(params[i, :]))
	meanVals = np.mean(params, axis=0)

	distSquares = []
	for point in pointsFromSingleConcept:
		distSquares.append(calcDistSq(point.indVars, meanVals))
	variance = m.pow(np.std(np.sqrt(distSquares)), 2)

	numSrcPoints = 0
	print("writing the third concept points")
	for i in range(len(distSquares)):
		prob = m.exp(-1 * (distSquares[i] / (2 * variance)))
		randNum = r.random()
		if randNum < prob and numSrcPoints < (numPointsThirdConcept * srcSplit):
			foutSrcFile.write(pointsFromSingleConcept[i].toString())
			numSrcPoints += 1
		else:
			foutTrgFile.write(pointsFromSingleConcept[i].toString())

	"""
	Now take points only from the fourth concept (only from third function)
	"""
	numPointsFourthConcept = 9000
	params = np.zeros((numPointsFourthConcept, numInd))
	pointsFromSingleConcept = []
	for j in range(numInd):
		if j == 0:
			params[:, j] = np.random.uniform(size=numPointsFourthConcept)
		elif j == 3:
			params[:, j] = np.random.uniform(low=0.7, high=1.0, size=numPointsFourthConcept)
		else:
			params[:, j] = np.random.uniform(low=0.0, high=0.3, size=numPointsFourthConcept)

	for i in range(numPointsFourthConcept):
		pointsFromSingleConcept.append(getYAccThirdConcept(params[i, :]))
	meanVals = np.mean(params, axis=0)

	distSquares = []
	for point in pointsFromSingleConcept:
		distSquares.append(calcDistSq(point.indVars, meanVals))
	variance = m.pow(np.std(np.sqrt(distSquares)), 2)

	numSrcPoints = 0
	print("writing the Fourth concept points")
	for i in range(len(distSquares)):
		prob = m.exp(-1 * (distSquares[i] / (2 * variance)))
		randNum = r.random()
		if randNum < prob and numSrcPoints < (numPointsFourthConcept * srcSplit):
			foutSrcFile.write(pointsFromSingleConcept[i].toString())
			numSrcPoints += 1
		else:
			foutTrgFile.write(pointsFromSingleConcept[i].toString())

	foutSrcFile.close()
	foutTrgFile.close()


genData('RegSynGlobalGradualDrift_source_streamhalf.csv', 'RegSynGlobalGradualDrift_target_streamhalf.csv', 5, 50000, 0.1)
