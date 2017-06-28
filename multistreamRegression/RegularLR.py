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
from Regression import Regression
import time
import sklearn.linear_model.LinearRegression as LinearRegression


class RUpdate(object):

    def __init__(self):
        super(RUpdate, self).__init__()

    @staticmethod
    def init_y_hat(src_path='RegSynGlobalGradualDrift_source_streamhalf.csv',
                   tgt_path='RegSynGlobalGradualDrift_target_streamhalf.csv', src_size=None, tgt_size=None):
        """
        input is: source dataset with y, here we assume it is a list of list, the name is source, target dataset with yhat, 
        here we assume it is a list of list, the name is target 
        """
        matrix_ = Regression.read_csv(src_path, src_size)   # matrix_ is source data
        print "Source data read completes"
        m_size, n_size = matrix_.shape

        fix_srcsize = 50
        fix_trgsize = 900
        predicty = []
        error = []
        target_ = Regression.read_csv(tgt_path, size=tgt_size)
        print "Target data read completes"
        true_y1 = RUpdate.get_true_y(target_)
        for i in range(0,m_size/100):
            print i,m_size

            linear_regression = LinearRegression.LinearRegressionBaseLine({
                "fit_intercept": True
            })
            linear_regression.train(matrix_[i*100: i*100 + fix_srcsize])
            #print 'target'
            #print target_[i*9000:i*9000+fix_trgsize,:-1]
            tmppredict_y = linear_regression.predict(target_[i*900:i*900+fix_trgsize,:-1]).transpose().tolist()[0]
            #print 'tmppredicty'
            #print tmppredict_y
            predicty = predicty + tmppredict_y
            tmperror = RUpdate.get_prediction_error(predicty, true_y1[:len(predicty)])
            error.append(tmperror)
            #print tmperror
        return predicty, target_, error


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

start_time = time.clock()
yhat, origtarget, sep_error = RUpdate.init_y_hat()
ru = RUpdate()

true_y = ru.get_true_y(origtarget)
error = ru.get_prediction_error(yhat,true_y)
end_time = time.clock()

print "Execution time for %d iterations is: %s min" % (
1000, (end_time-start_time)/60.0)
#print "update y hat pm2.5", updateYhat
print "pmupdate",error
print sep_error