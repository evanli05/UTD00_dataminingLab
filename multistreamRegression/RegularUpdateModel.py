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
#from Regression import read_csv
import Regression
import time
from Regression import Regression



class RegularUpdate(object):

    def __init__(self):
        super(RegularUpdate, self).__init__()

    @staticmethod
    def init_y_hat(src_path='C:/Users/sk/OneDrive - The University of Texas at Dallas/Workspace/UTD00_dataminingLab/multistreamRegression/pm2.5_srcFile.csv',
                   tgt_path='C:/Users/sk/OneDrive - The University of Texas at Dallas/Workspace/UTD00_dataminingLab/multistreamRegression/pm2.5_trgFile.csv', src_size=1000, stopThd = 1e-5, rateInitial = 0.01,
                   decayTune = 0.01, iteration = 2000, tgt_size=9000):
        """
        input is: source dataset with y, here we assume it is a list of list, the name is source, target dataset with yhat, 
        here we assume it is a list of list, the name is target 
        """
        matrix_ = Regression.read_csv(src_path, src_size)   # matrix_ is source data
#        matrix_ = Regression.read_csv(src_path)   # matrix_ is source data
        m_size, n_size = matrix_.shape
        m0_ = np.matrix(np.zeros(1 + n_size), dtype=np.float64)


        fix_srcsize = 200
        fix_trgsize = 1800
        predicty = []
        error = []
        target_ = Regression.read_csv(tgt_path, size=tgt_size)
        true_y1 = RegularUpdate.get_true_y(target_)
        for i in range(0,m_size/200):
            print i
            best_m_ = Regression.start(
                m0_, matrix_[i*200: i*200 + fix_srcsize], stop_thd=stopThd, rate_initial=rateInitial, decay_tune=decayTune, iteration=iteration)
            print best_m_
            #target_ = Regression.read_csv(tgt_path, size=tgt_size)
            _,tmp_predict_y = Regression.get_est_mean(best_m_, target_[i*1800:i*1800+fix_trgsize])
            predicty = predicty + tmp_predict_y
            tmperror = RegularUpdate.get_prediction_error(predicty, true_y1[:len(predicty)])
            error.append(tmperror)
            print tmperror
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
yhat, origtarget, sep_error = RegularUpdate.init_y_hat()
#start_time = time.clock()
ru = RegularUpdate()

true_y = ru.get_true_y(origtarget)
error = ru.get_prediction_error(yhat,true_y)
end_time = time.clock()

print "Execution time for %d iterations is: %s min" % (
500, (end_time-start_time)/60.0)
#print "update y hat pm2.5", updateYhat
print "localabrupterror_200_5",error
print sep_error