import numpy as np
import scipy
import math, numpy as np
from scipy.stats import beta, binom
#import matplotlib.pyplot as plt
from decimal import Decimal
import sys
from operator import truediv
#from mlab.releases import R2013a as mlab
from random import randint
import random
import csv
from Regression1 import Regression
import time


class ChangeDetection:
    CUSHION = 30
    def __init__(self, gamma, sensitivity, maxWindowSize):
        self.gamma = gamma
        self.sensitivity = sensitivity
        self.maxWindowSize = maxWindowSize

    """
    	 Functions to estimate beta distribution parameters
    """

    def __calcBetaDistAlpha(self, list, sampleMean, sampleVar):
        if sampleMean == -1:
            sampleMean = np.mean(list)
        if sampleVar == -1:
            sampleVar = np.var(list)

        c = (sampleMean * (1 - sampleMean) / sampleVar) - 1

        return sampleMean * c

    def __calcBetaDistBeta(self, list, alphaChange, sampleMean):
        if sampleMean == -1:
            sampleMean = np.mean(list)
        return alphaChange * ((1.0 / sampleMean) - 1)

    """
    	input: The dynamic sliding window containing confidence of target classifier
        output: -1 if no change found, otherwise the change point
    """
    def detectChange(self, slidingWindow):
        estimatedChangePoint = -1
        N = len(slidingWindow)
        cushion = max(self.CUSHION, int(math.floor(N ** self.gamma)))

        # If mean confidence fall below 0.3, must retrain the classifier, so return a changepoint
        if N > self.maxWindowSize:
                print('Current target Window Size is: ' + str(N) + 'it exceed the max window size')
                return 0
        if N > 2 * cushion and np.mean(slidingWindow[0:N]) <= 0.3:
            print('Current target Window Size is: ' + str(N))
            print('But overall confidence fell below 0.3, so update model' )
            return 0

        threshold = 1
        #threshold = -math.log(self.sensitivity)
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
                alphaPreChange = self.__calcBetaDistAlpha(slidingWindow[:k], xbar0, var0)
                betaPreChange = self.__calcBetaDistBeta(slidingWindow[:k], alphaPreChange, xbar0)
                alphaPostChange = self.__calcBetaDistAlpha(slidingWindow[k:], xbar1, var1)
                betaPostChange = self.__calcBetaDistBeta(slidingWindow[k:], alphaPostChange, xbar1)

                try:
                    swin = map(float, slidingWindow[k:])
                    #swin1 =map(float, slidingWindow[:k])
                    denom = [beta.pdf(s, alphaPreChange, betaPreChange) for s in swin]
                    nor_denom = np.array([0.001 if h == 0 else h for h in denom])
                    nor_swin = swin / nor_denom
                    #print 'nor_swin'
                    #print nor_swin
                    skn = sum([Decimal(beta.pdf(ns, alphaPostChange, betaPostChange)) for ns in nor_swin])
                    #print 'skn'
                    #print skn
                except:
                    e = sys.exc_info()
                    print str(e[1])
                    raise Exception('Error in calculating skn')

                if skn > w:

                    w = skn
                    print w
                    kAtMaxW = k

        if w >= threshold and kAtMaxW != -1:
            print w, threshold
            estimatedChangePoint = kAtMaxW
            print('Estimated change point is ' + str(estimatedChangePoint) + ', detect at ' + str(N) )
        return estimatedChangePoint


class Update(object):

    def __init__(self):
        super(Update, self).__init__()
        self.change_detection = ChangeDetection(gamma=0.5, maxWindowSize=1000, sensitivity=0.05)

    sourceList=[]
    targetList=[]

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
        return m, m-h, m+h

    @staticmethod
    def sigmoid(x):
        """
        calculate sigmoid of x
        :param x: a number
        :return: 
        """
        return 1 / (1+math.exp(-x))

    @staticmethod
    def getconf(targetList):
        conf = []
        if len(targetList)>50:
            for i in range(0,len(targetList)-50):
                d = targetList[i: i+50]
                dd = Update.mean_confidence_interval(d, confidence=0.95)
                conf.append(Update.sigmoid(dd[2]-dd[1]))
        return conf

    @staticmethod
    def init_y_hat(source_matrix=None, target_matrix=None, src_path='C:\Users\sk\OneDrive - The University of Texas at Dallas\Workspace\UTD00_dataminingLab\multistreamRegression\pm2.5_srcFile.csv',
                   tgt_path='C:\Users\sk\OneDrive - The University of Texas at Dallas\Workspace\UTD00_dataminingLab\multistreamRegression\pm2.5_trgFile.csv', src_size=500, stopThd = 1e-5, rateInitial = 0.01,
                   decayTune = 0.01, iteration = 200000, tgt_size=None):
        """
        input is: source dataset with y, here we assume it is a list of list, the name is source, target dataset with yhat, 
        here we assume it is a list of list, the name is target 
        """
        if source_matrix is None:
            source_matrix_ = Regression.read_csv(src_path, None)   # matrix_ is source data
        else:
            source_matrix_ = source_matrix
        matrix_ = source_matrix_[:src_size, :]

        if target_matrix is None:
            target_ = Regression.read_csv(tgt_path, size=tgt_size)
        else:
            target_ = target_matrix

        return source_matrix_, target_

    def UpdateModel(self, source, target, stopThd = 1e-5, rateInitial = 0.01, decayTune = 0.01, iteration = 1000):

        yhat = []
        tempsource = []
        temptarget = []
        sourceIndex = 0
        targetIndex = 0
        src_count = 0
        tgt_count = 0
        src_size, _ = source.shape
        tgt_size, _ = target.shape
        window_size_threshold = 200
        last_size = window_size_threshold

        #print "update model", src_size, source.shape
        while True:
            if src_count >= src_size or tgt_count >= tgt_size:
                break

            #print "test 1", src_count, tgt_count, src_size, tgt_size
            #data_type=random.uniform(0,1)
            #if data_type<0.1139:
            data_type=randint(1,10)
            if data_type<2:
                print("get data from source")
                tempsource.append(source[sourceIndex])
                sourceIndex+=1
                src_count += 1
                print src_count
            else:
                print("get data from target")
                temptarget.append(target[targetIndex])
                targetIndex+=1
                tgt_count += 1
                print tgt_count
            print "temptarget", len(temptarget)
            if len(temptarget)>window_size_threshold:
                if len(temptarget) - last_size < 100:
                    continue
                else:
                    last_size = len(temptarget)
                    matrix_ = source[sourceIndex-len(tempsource)+1:sourceIndex]
                    m_size, n_size = matrix_.shape
                    m0_ = np.matrix(np.zeros(1 + n_size), dtype=np.float64)
                    best_m_ = Regression.start(
                        m0_, matrix_,target[targetIndex-len(temptarget)+1:targetIndex], stop_thd=stopThd, rate_initial=rateInitial, decay_tune=decayTune,
                        iteration=iteration)
                    currentyhat = Regression.get_predictY(best_m_, matrix_,target[targetIndex-len(temptarget)+1:targetIndex])

                    currentconf_yhat = Update.getconf(currentyhat)


                slidingwindow = currentconf_yhat
                changeIndex = self.change_detection.detectChange(slidingwindow)
                print "check2"
                # if distribution changes, return current index, else -1
                if changeIndex != -1:
                    index = changeIndex +targetIndex -len(temptarget)+1

                    updatetarget = target[index:targetIndex, :]
                    updatetarget0 = target[targetIndex - len(temptarget) + 1:index, :]
                    ratio = (changeIndex + 1)/float(len(temptarget))
                    src_change_index = int(math.ceil(len(tempsource) * ratio))
                    indexsrc = src_change_index + sourceIndex - len(tempsource) + 1

                    if sourceIndex-indexsrc>20:
                        updatesource = source[indexsrc:sourceIndex]
                        tmp = None
                        for each_row in updatesource:
                            if tmp is None:
                                tmp = each_row
                            else:
                                tmp = np.vstack((tmp, each_row))
                        updatesource = tmp
                        m_size ,n_size = source.shape
                        m0_ = np.matrix(np.zeros(1 + n_size), dtype=np.float64)

                        best_m1_ = Regression.start(
                            m0_, updatesource, updatetarget, stop_thd=stopThd, rate_initial=rateInitial,
                            decay_tune=decayTune, iteration=iteration)
                        print "Re-training1 Complete!"
                        predict_y1 = Regression.get_predictY(best_m1_, updatesource, updatetarget)
                        if indexsrc -sourceIndex+len(tempsource)-1>20:
                            updatesource0 = source[sourceIndex - len(tempsource) + 1:indexsrc, :]
                            tmp0 = None
                            for each_row0 in updatesource0:
                                if tmp0 is None:
                                    tmp0 = each_row0
                                else:
                                    tmp0 = np.vstack((tmp0, each_row0))
                            updatesource0 = tmp0

                            m00_ = np.matrix(np.zeros(1 + n_size), dtype=np.float64)

                            best_m10_ = Regression.start(
                                m00_, updatesource0, updatetarget0, stop_thd=stopThd, rate_initial=rateInitial,
                                decay_tune=decayTune, iteration=iteration)
                            print "Re-training0 Complete!"
                            predict_y0 = Regression.get_predictY(best_m10_, updatesource0, updatetarget0)
                            for p in range(0, changeIndex):
                                currentyhat[p] = predict_y0[p]
                        for k in range(changeIndex, len(currentyhat)):
                            currentyhat[k] = predict_y1[k - changeIndex]


                        yhat = yhat + currentyhat
                        true_y1 = Update.get_true_y(Update.init_y_hat()[1])
                        tmptrue_y1 = true_y1[:len(yhat)]
                        tmpyhat = yhat
                        print 'tmpyhatlen'
                        print len(tmpyhat)
                        tmperror1 = Update.get_prediction_error(tmpyhat, tmptrue_y1)

                        with open('error_19synpm1000.csv','a+') as f:
                            writer = csv.writer(f)
                            writer.writerow([index, tmperror1])
                        tempsource = []
                        temptarget = []
        return yhat

    @staticmethod
    def get_prediction_error(prediction_value, true_value):
        error = 0
        for i in range(len(prediction_value)):
            error += abs(prediction_value[i] - true_value[i])
        return error / len(prediction_value)

    @staticmethod
    def get_true_y(target):
        y = target[:, -1]
        return y.transpose().tolist()[0]


source, origtarget = Update.init_y_hat(src_size=500)
start_time = time.clock()
up = Update()
updateYhat = up.UpdateModel(source,origtarget)
true_y = up.get_true_y(origtarget)
error = up.get_prediction_error(updateYhat,true_y)
end_time = time.clock()

print "Execution time for %d iterations is: %s min" % (
1000, (end_time-start_time)/60.0)
#print "update y hat pm2.5", updateYhat
print "error syn19pm2.5",error